/**
 * Streaming proxy to FastAPI chat endpoint.
 * Next.js rewrites buffer SSE responses; this route streams the response
 * directly to the client for real-time updates.
 */
import { NextRequest } from "next/server";

const FASTAPI_URL =
  process.env.FASTAPI_URL ?? "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  const url = new URL("/api/chat", FASTAPI_URL);
  url.search = request.nextUrl.searchParams.toString();

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      "Content-Type": request.headers.get("Content-Type") ?? "application/json",
    },
    body: request.body,
    duplex: "half",
  } as RequestInit);

  if (!response.body) {
    return new Response("No response body", { status: 502 });
  }

  const headers = new Headers();
  const forwardHeaders = [
    "content-type",
    "cache-control",
    "connection",
    "x-vercel-ai-ui-message-stream",
    "x-vercel-ai-protocol",
    "x-accel-buffering",
  ];
  for (const name of forwardHeaders) {
    const value = response.headers.get(name);
    if (value) headers.set(name, value);
  }
  if (!headers.has("x-accel-buffering")) headers.set("x-accel-buffering", "no");

  return new Response(response.body, {
    status: response.status,
    headers,
  });
}
