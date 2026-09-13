import type { NextConfig } from "next"

// The browser only talks to Next.js. /api/* is proxied to the Python API
// (face-recognition-system/api.py), so no CORS configuration is needed.
const FACE_API_URL = process.env.FACE_API_URL ?? "http://127.0.0.1:8000"

const nextConfig: NextConfig = {
  // Hide the on-screen Next.js dev badge (errors are still shown during development).
  devIndicators: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${FACE_API_URL}/api/:path*` }]
  },
}

export default nextConfig
