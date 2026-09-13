// Typed client for the Python API (face-recognition-system/api.py), reached via
// the /api rewrite in next.config.ts.

export type SystemStatus = {
  enrolled_count: number
  confirmed_threshold: number
  uncertain_lower_bound: number
  max_identification_attempts: number
  min_enrollment_images: number
  max_enrollment_images: number
  model_pack: string
  evaluation_threshold_max: number
}

export type Person = {
  person_id: number
  name: string
  num_samples: number
  created_at: string
  has_thumbnail: boolean
}

export type MatchStatus = "confirmed" | "uncertain" | "unknown"

export type FaceResult = {
  index: number
  status: MatchStatus
  person_name: string | null
  similarity: number
}

export type IdentifyResponse = {
  faces: FaceResult[]
  annotated_image: string | null
  /** Confirmed threshold actually applied by the backend (evaluation override or the default). */
  confirmed_threshold: number
  default_confirmed_threshold: number
  uncertain_lower_bound: number
}

export type EnrollResponse = {
  person_id: number
  name: string
  num_samples: number
  updated: boolean
  files: { filename: string; face_count: number; accepted: boolean }[]
}

export class ApiError extends Error {}

const API_DOWN_MESSAGE =
  "Could not reach the recognition API. Start it from face-recognition-system with: uvicorn api:app --port 8000"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, { cache: "no-store", ...init })
  } catch {
    throw new ApiError(API_DOWN_MESSAGE)
  }
  if (!res.ok) {
    let detail: string | undefined
    try {
      const body = await res.json()
      if (typeof body.detail === "string") detail = body.detail
    } catch {
      // Non-JSON error: the Next.js proxy could not reach the Python API.
    }
    throw new ApiError(detail ?? (res.status >= 500 ? API_DOWN_MESSAGE : `Request failed (${res.status})`))
  }
  return res.json() as Promise<T>
}

function formWith(entries: [string, string | File][]) {
  const form = new FormData()
  for (const [key, value] of entries) form.append(key, value)
  return form
}

export const api = {
  status: () => request<SystemStatus>("/api/status"),
  persons: () => request<Person[]>("/api/persons"),
  deletePerson: (personId: number) =>
    request<{ deleted: number }>(`/api/persons/${personId}`, { method: "DELETE" }),
  detect: (file: File) =>
    request<{ filename: string; face_count: number }>("/api/detect", {
      method: "POST",
      body: formWith([["file", file]]),
    }),
  enroll: (name: string, files: File[]) =>
    request<EnrollResponse>("/api/enroll", {
      method: "POST",
      body: formWith([["name", name], ...files.map((f): [string, File] => ["files", f])]),
    }),
  // confirmedThreshold: evaluation/testing override. Omitted => backend uses the configured default.
  identify: (file: File, confirmedThreshold?: number) =>
    request<IdentifyResponse>("/api/identify", {
      method: "POST",
      body: formWith([
        ["file", file],
        ...(confirmedThreshold === undefined
          ? []
          : [["confirmed_threshold", confirmedThreshold.toFixed(2)] as [string, string]]),
      ]),
    }),
}

export function thumbnailUrl(person: Person) {
  // created_at changes on re-enrollment, so the browser won't show a stale thumbnail.
  return `/api/persons/${person.person_id}/thumbnail?v=${encodeURIComponent(person.created_at)}`
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}
