// k6 load test for the edge (Cloudflare Worker) or a single origin.
//
//   k6 run -e BASE_URL=https://edge.example.com -e API_KEY=fa_... loadtest/k6.js
//   k6 run -e SCENARIO=smoke ...      # 1 user for 1 minute (used after deploys)
//
// 429 responses are expected under load: they are admission control working
// (shed load with Retry-After) rather than errors, so they are tracked separately
// from failures. Use unique questions (CACHE_BUST=1) to measure the model rather
// than the edge cache.

import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const API_KEY = __ENV.API_KEY || "";
const SCENARIO = __ENV.SCENARIO || "load";
const CACHE_BUST = __ENV.CACHE_BUST === "1";

const shed = new Counter("shed_429");
const failures = new Rate("failures");
const cacheHits = new Rate("edge_cache_hits");
const tokens = new Trend("completion_tokens");

const QUESTIONS = [
  "How do I declare a required query parameter in FastAPI?",
  "How do I return a 201 status code from a POST endpoint?",
  "How do I use Depends with a class-based dependency?",
  "How do I add CORS middleware for a single frontend origin?",
  "How do I stream a large file with StreamingResponse?",
  "How do I test a FastAPI app with TestClient?",
];

const scenarios = {
  smoke: { executor: "constant-vus", vus: 1, duration: "1m" },
  load: {
    executor: "ramping-arrival-rate",
    startRate: 1,
    timeUnit: "1m",
    preAllocatedVUs: 10,
    maxVUs: 30,
    stages: [
      { target: 6, duration: "2m" },
      { target: 20, duration: "5m" },
      { target: 0, duration: "1m" },
    ],
  },
};

export const options = {
  scenarios: { [SCENARIO]: scenarios[SCENARIO] },
  thresholds: {
    failures: ["rate<0.02"],
    "http_req_duration{expected_response:true}": ["p(95)<60000"],
  },
};

export default function () {
  const base = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
  const question = CACHE_BUST ? `${base} (${__VU}-${__ITER}-${Date.now()})` : base;
  const res = http.post(
    `${BASE_URL}/generate`,
    JSON.stringify({ question, max_new_tokens: 128, temperature: 0 }),
    {
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      timeout: "120s",
      responseCallback: http.expectedStatuses(200, 429),
    },
  );

  if (res.status === 429) {
    shed.add(1);
    const retry = Number(res.headers["Retry-After"] || 5);
    sleep(Math.min(retry, 30));
    return;
  }
  const ok = check(res, {
    "status is 200": (r) => r.status === 200,
    "has an answer": (r) => r.status === 200 && r.json("answer").length > 0,
  });
  failures.add(!ok);
  if (ok) {
    cacheHits.add(res.headers["X-Cache"] === "HIT");
    tokens.add(res.json("usage.completion_tokens"));
  }
  sleep(1);
}
