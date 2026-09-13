"use client"

import * as React from "react"
import { api, errorMessage, type SystemStatus } from "@/lib/api"

// Thresholds and limits come from src/config.py via the API — never duplicated here.
export function useSystemStatus() {
  const [status, setStatus] = React.useState<SystemStatus | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(() => {
    api
      .status()
      .then((s) => {
        setStatus(s)
        setError(null)
      })
      .catch((e) => setError(errorMessage(e)))
  }, [])

  React.useEffect(refresh, [refresh])

  return { status, error, refresh }
}
