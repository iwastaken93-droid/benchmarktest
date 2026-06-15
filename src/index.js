// Cloudflare Worker entry point
// Serves static assets, triggers GitHub Actions hourly, and exposes benchmark trigger/status endpoints.

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // Endpoint: POST /v1/benchmark/run (Manual trigger)
    if (url.pathname === '/v1/benchmark/run' && request.method === 'POST') {
      try {
        await triggerGitHubAction(env);
        return new Response(JSON.stringify({ 
          success: true, 
          message: "Manual benchmark run initiated on GitHub Actions." 
        }), {
          headers: { 'Content-Type': 'application/json' }
        });
      } catch (e) {
        return new Response(JSON.stringify({ 
          success: false, 
          message: `Failed to trigger benchmark: ${e.message}` 
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    // Endpoint: GET /v1/benchmark/status (Polling runner status)
    if (url.pathname === '/v1/benchmark/status' && request.method === 'GET') {
      const status = await checkGitHubWorkflowStatus(env);
      return new Response(JSON.stringify(status), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Endpoint: GET /benchmark_results.json (or /public/benchmark_results.json)
    // Fetch directly from GitHub to bypass Cloudflare Pages deployment delay
    if ((url.pathname === '/benchmark_results.json' || url.pathname === '/public/benchmark_results.json') && request.method === 'GET') {
      try {
        const repo = env.GITHUB_REPO || 'iwastaken93-droid/benchmarktest';
        const ref = env.GITHUB_REF || 'master';
        const rawUrl = `https://raw.githubusercontent.com/${repo}/${ref}/public/benchmark_results.json?t=${Date.now()}`;
        
        const response = await fetch(rawUrl, {
          headers: {
            'User-Agent': 'cloudflare-worker-results-proxy'
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify(data), {
            headers: { 
              'Content-Type': 'application/json',
              'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate'
            }
          });
        }
      } catch (e) {
        // Fallback to static asset serving if github fetch fails
        console.error("Failed to fetch live results from GitHub, falling back to static asset:", e);
      }
    }

    // Fallback: serve static frontend dashboard/results assets from CDN
    return env.ASSETS.fetch(request);
  },

  async scheduled(event, env, ctx) {
    ctx.waitUntil(
      triggerGitHubAction(env)
        .then(() => console.log("Hourly benchmark trigger successful."))
        .catch(err => console.error("Hourly benchmark trigger failed:", err))
    );
  }
};

// Trigger a workflow_dispatch run on GitHub Actions
async function triggerGitHubAction(env) {
  const repo = env.GITHUB_REPO || 'iwastaken93-droid/benchmarktest';
  const workflow = env.GITHUB_WORKFLOW || 'run_benchmark.yml';
  const ref = env.GITHUB_REF || 'master';
  const token = env.GITHUB_TOKEN;

  if (!token) {
    throw new Error("GITHUB_TOKEN secret is not configured in Cloudflare Worker environment variables.");
  }

  const url = `https://api.github.com/repos/${repo}/actions/workflows/${workflow}/dispatches`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'cloudflare-cron-trigger-worker',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ ref })
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`GitHub API Error (${response.status}): ${text}`);
  }
  return true;
}

// Queries GitHub Actions API for active workflow runs
async function checkGitHubWorkflowStatus(env) {
  const repo = env.GITHUB_REPO || 'iwastaken93-droid/benchmarktest';
  const workflow = env.GITHUB_WORKFLOW || 'run_benchmark.yml';
  const token = env.GITHUB_TOKEN;

  if (!token) {
    // Graceful fallback to help users diagnose missing credentials
    return { running: false, status: "Idle (GITHUB_TOKEN not configured on Cloudflare)" };
  }

  try {
    const url = `https://api.github.com/repos/${repo}/actions/workflows/${workflow}/runs?per_page=1`;
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'cloudflare-cron-trigger-worker'
      }
    });

    if (!response.ok) {
      const text = await response.text();
      return { running: false, status: `Idle (GitHub API Error: ${response.status})` };
    }

    const data = await response.json();
    const latestRun = data.workflow_runs?.[0];
    
    if (latestRun) {
      const runningStatuses = ['queued', 'in_progress', 'waiting', 'requested'];
      if (runningStatuses.includes(latestRun.status)) {
        return { 
          running: true, 
          status: `Running on GitHub Actions (Run #${latestRun.run_number})` 
        };
      }
    }
    return { running: false, status: "Idle" };
  } catch (e) {
    return { running: false, status: `Idle (Error checking status: ${e.message})` };
  }
}
