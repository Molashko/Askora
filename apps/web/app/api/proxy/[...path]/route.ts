import { NextRequest, NextResponse } from "next/server";

/** Large CSV uploads/imports may legitimately run for many minutes. */
const UPSTREAM_TIMEOUT_MS = 3_600_000;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

function upstreamBase(): string {
  return (process.env.INTERNAL_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
}

async function forward(req: NextRequest, pathSegments: string[]): Promise<NextResponse> {
  const suffix = pathSegments.length ? pathSegments.join("/") : "";
  const upstreamUrl = `${upstreamBase()}/api/v1/${suffix}${req.nextUrl.search}`;

  const headers = new Headers();
  for (const name of ["cookie", "content-type", "accept", "accept-language", "authorization"]) {
    const v = req.headers.get(name);
    if (v) {
      headers.set(name, v);
    }
  }

  const init: RequestInit = {
    method: req.method,
    headers,
    cache: "no-store",
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    if (req.body) {
      init.body = req.body;
      (init as RequestInit & { duplex: "half" }).duplex = "half";
    }
  }

  const res = await fetch(upstreamUrl, init);
  const body = await res.arrayBuffer();
  const out = new NextResponse(body, { status: res.status, statusText: res.statusText });

  const ct = res.headers.get("content-type");
  if (ct) {
    out.headers.set("content-type", ct);
  }
  const cc = res.headers.get("cache-control");
  if (cc) {
    out.headers.set("cache-control", cc);
  }
  const rid = res.headers.get("x-request-id");
  if (rid) {
    out.headers.set("x-request-id", rid);
  }

  const getSetCookie = res.headers.getSetCookie?.bind(res.headers);
  if (getSetCookie) {
    for (const c of getSetCookie()) {
      out.headers.append("set-cookie", c);
    }
  } else {
    const sc = res.headers.get("set-cookie");
    if (sc) {
      out.headers.append("set-cookie", sc);
    }
  }

  return out;
}

type RouteCtx = { params: { path: string[] } };

export async function GET(req: NextRequest, ctx: RouteCtx) {
  return forward(req, ctx.params.path ?? []);
}

export async function HEAD(req: NextRequest, ctx: RouteCtx) {
  return forward(req, ctx.params.path ?? []);
}

export async function POST(req: NextRequest, ctx: RouteCtx) {
  return forward(req, ctx.params.path ?? []);
}

export async function PUT(req: NextRequest, ctx: RouteCtx) {
  return forward(req, ctx.params.path ?? []);
}

export async function PATCH(req: NextRequest, ctx: RouteCtx) {
  return forward(req, ctx.params.path ?? []);
}

export async function DELETE(req: NextRequest, ctx: RouteCtx) {
  return forward(req, ctx.params.path ?? []);
}
