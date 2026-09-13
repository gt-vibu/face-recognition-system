"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import {
  ArrowRight,
  Check,
  ChevronDown,
  CircleX,
  LoaderCircle,
  Plus,
  RotateCcw,
  SlidersHorizontal,
  TriangleAlert,
} from "lucide-react"

import { PageHeader } from "@/components/page-header"
import { PersonAvatar } from "@/components/person-avatar"
import { SimilarityMeter } from "@/components/similarity-meter"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button, buttonVariants } from "@/components/ui/button"
import { useSystemStatus } from "@/hooks/use-system-status"
import { api, errorMessage, type FaceResult, type IdentifyResponse, type MatchStatus, type Person } from "@/lib/api"
import { cn } from "@/lib/utils"

const ACCEPTED = ["image/jpeg", "image/png"]

/*
 * Bounded retry for Uncertain matches, kept in the browser (same behaviour as the
 * Streamlit page's session_state):
 *   - each newly uploaded photo that yields an Uncertain face counts as ONE attempt
 *     (counting happens once, in the upload handler — re-renders never count);
 *   - a photo with no Uncertain face (Confirmed or Unknown) resets the counter;
 *   - a photo with no detectable face does not count;
 *   - reaching the maximum locks the flow until "Try Again".
 */
export default function IdentifyPage() {
  const router = useRouter()
  const { status, error: statusError } = useSystemStatus()
  const maxAttempts = status?.max_identification_attempts ?? 3

  const [attempt, setAttempt] = React.useState(0)
  const [locked, setLocked] = React.useState(false)
  const [busy, setBusy] = React.useState(false)
  const [result, setResult] = React.useState<IdentifyResponse | null>(null)
  const [requestError, setRequestError] = React.useState<string | null>(null)
  // Evaluation/testing override of the Confirmed threshold for this tab; null = configured default.
  // It is sent to the backend with each upload — the decision itself is always made server-side.
  const [evalThreshold, setEvalThreshold] = React.useState<number | null>(null)

  // Only used to show enrolled people's thumbnails next to results (names are unique).
  const [people, setPeople] = React.useState<Person[]>([])
  React.useEffect(() => {
    api.persons().then(setPeople, () => {})
  }, [])
  const personByName = React.useMemo(() => new Map(people.map((p) => [p.name, p])), [people])

  function resetSession() {
    setAttempt(0)
    setLocked(false)
    setResult(null)
    setRequestError(null)
  }

  async function handleUpload(file: File) {
    setBusy(true)
    setRequestError(null)
    try {
      const res = await api.identify(file, evalThreshold ?? undefined)
      setResult(res)
      if (res.faces.length === 0) return // "No face detected" — not an attempt
      if (res.faces.some((f) => f.status === "uncertain")) {
        const next = attempt + 1
        setAttempt(next)
        if (next >= maxAttempts) setLocked(true)
      } else {
        setAttempt(0)
      }
    } catch (e) {
      setRequestError(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  function goToEnrollment() {
    resetSession()
    router.push("/enroll")
  }

  // Same drop handling as the Enroll panel; one photo per attempt.
  const [dragging, setDragging] = React.useState(false)
  const [wrongType, setWrongType] = React.useState(false)
  const dragDepth = React.useRef(0)
  function takeFiles(files: File[]) {
    const image = files.find((f) => ACCEPTED.includes(f.type))
    setWrongType(!image && files.length > 0)
    if (image) handleUpload(image)
  }

  const anyUncertain = result?.faces.some((f) => f.status === "uncertain") ?? false
  const remaining = maxAttempts - attempt

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 sm:pt-2">
      <PageHeader title="Identify a face" description="Upload a photo. Each face is matched against everyone enrolled." />

      {statusError && (
        <Alert variant="destructive" className="border-red-200 bg-red-50">
          <CircleX />
          <AlertTitle>Recognition API not reachable</AlertTitle>
          <AlertDescription>{statusError}</AlertDescription>
        </Alert>
      )}

      <section
        aria-label="Identification"
        className={cn(
          "rounded-lg border border-border bg-white p-5 shadow-[0_1px_2px_rgb(0_0_0/0.03)] transition-[border-color,box-shadow] sm:p-6",
          dragging && "border-brand ring-3 ring-brand/15"
        )}
        onDragEnter={(e) => {
          e.preventDefault()
          dragDepth.current += 1
          setDragging(true)
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1)
          if (dragDepth.current === 0) setDragging(false)
        }}
        onDrop={(e) => {
          e.preventDefault()
          dragDepth.current = 0
          setDragging(false)
          if (!busy && !locked) takeFiles(Array.from(e.dataTransfer.files))
        }}
      >
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-medium">Photo</h2>
          <span className={cn("text-sm tabular-nums", attempt > 0 ? "text-amber-700" : "text-muted-foreground")}>
            {locked ? `${maxAttempts} of ${maxAttempts} attempts used` : `Attempt ${attempt + 1} of ${maxAttempts}`}
          </span>
        </div>
        {evalThreshold !== null && status && (
          <p className="mt-1 text-xs text-amber-700">
            Evaluation override active: confirmed threshold {evalThreshold.toFixed(2)} (default{" "}
            {status.confirmed_threshold.toFixed(2)})
          </p>
        )}

        {locked ? (
          <LockedState maxAttempts={maxAttempts} onTryAgain={resetSession} onEnroll={goToEnrollment} />
        ) : (
          <>
            <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
              <label
                htmlFor="identify-photo"
                className={cn(
                  buttonVariants({ variant: "outline", size: "sm" }),
                  "cursor-pointer",
                  busy && "pointer-events-none opacity-50"
                )}
              >
                <Plus /> {result ? "Upload another photo" : "Upload photo"}
              </label>
              <span className="text-xs text-muted-foreground">
                {dragging ? "Drop to identify" : "JPG or PNG · Group photos work too"}
              </span>
              {wrongType && <span className="text-xs text-red-700">Only JPG or PNG images are supported</span>}
            </div>
            <input
              id="identify-photo"
              type="file"
              accept={ACCEPTED.join(",")}
              disabled={busy}
              className="sr-only"
              onChange={(e) => {
                const files = Array.from(e.target.files ?? [])
                e.target.value = "" // allow re-selecting the same file as a new attempt
                takeFiles(files)
              }}
            />

            {requestError && <p className="mt-4 text-sm text-red-700">{requestError}</p>}

            {!result ? (
              <label
                htmlFor="identify-photo"
                className={cn(
                  "mt-4 flex min-h-56 cursor-pointer flex-col items-center justify-center gap-1 rounded-md border border-dashed border-stone-300 text-muted-foreground transition-colors hover:border-stone-400 hover:bg-stone-50 hover:text-foreground",
                  busy && "pointer-events-none"
                )}
              >
                {busy ? <LoaderCircle className="size-4 animate-spin" /> : <Plus className="size-4" />}
                <span className="text-sm">{busy ? "Analysing photo…" : "Add a photo to identify"}</span>
              </label>
            ) : result.faces.length > 1 ? (
              // Group photo: summary, the full-width photo with a legend, then one card per face.
              <div className={cn("mt-4 flex flex-col gap-4", busy && "opacity-60")}>
                <FaceSummary faces={result.faces} />
                {result.annotated_image && (
                  <figure className="flex flex-col gap-2">
                    <AnnotatedPhoto src={result.annotated_image} />
                    <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      {(Object.keys(tones) as MatchStatus[]).map((s) => (
                        <span key={s} className="flex items-center gap-1.5">
                          <span className={cn("size-2 rounded-full", tones[s].dot)} /> {tones[s].label}
                        </span>
                      ))}
                      <span>Faces are numbered left to right</span>
                    </figcaption>
                  </figure>
                )}
                <div className="grid gap-3 sm:grid-cols-2">
                  {result.faces.map((face) => (
                    <div key={face.index} className={cn("rounded-md border border-border border-t-[3px] p-4", tones[face.status].edge)}>
                      <FaceResultRow
                        face={face}
                        showIndex
                        faceImage={face.face_image}
                        person={face.person_name ? personByName.get(face.person_name) : undefined}
                        lower={status?.uncertain_lower_bound}
                        confirmed={result.confirmed_threshold}
                        onEnroll={goToEnrollment}
                      />
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className={cn("mt-4 grid items-start gap-5 md:grid-cols-2", busy && "opacity-60")}>
                {result.annotated_image && <AnnotatedPhoto src={result.annotated_image} />}
                <div className="flex flex-col divide-y divide-border">
                  {result.faces.length === 0 ? (
                    <p className="flex items-center gap-1.5 text-sm text-red-700">
                      <CircleX className="size-4" /> No face detected. Please upload a clearer image.
                    </p>
                  ) : (
                    result.faces.map((face) => (
                      <FaceResultRow
                        key={face.index}
                        face={face}
                        showIndex={result.faces.length > 1}
                        person={face.person_name ? personByName.get(face.person_name) : undefined}
                        lower={status?.uncertain_lower_bound}
                        confirmed={result.confirmed_threshold}
                        onEnroll={goToEnrollment}
                      />
                    ))
                  )}
                </div>
              </div>
            )}

            {anyUncertain && !busy && (
              <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-5">
                <p className="text-sm text-amber-700">
                  Please upload another clear image · {remaining} attempt(s) remaining.
                </p>
                <label htmlFor="identify-photo" className={cn(buttonVariants({ size: "lg" }), "cursor-pointer px-4")}>
                  Upload another photo <ArrowRight />
                </label>
              </div>
            )}
          </>
        )}

        {status && (
          <details className="group mt-5 border-t border-border pt-4">
            <summary className="flex cursor-pointer list-none items-center gap-2.5 rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground transition-colors select-none hover:bg-muted focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none [&::-webkit-details-marker]:hidden">
              <SlidersHorizontal className="size-4 shrink-0 text-muted-foreground" />
              <span className="flex-1">Evaluation threshold — testing only</span>
              <span
                className={cn(
                  "tabular-nums",
                  evalThreshold !== null ? "font-medium text-amber-700" : "text-muted-foreground",
                )}
              >
                {(evalThreshold ?? status.confirmed_threshold).toFixed(2)}
                {evalThreshold === null && " (default)"}
              </span>
              <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
            </summary>
            <div className="mt-3 grid gap-2 px-1">
              <input
                type="range"
                aria-label="Evaluation confirmed-match similarity threshold"
                min={status.uncertain_lower_bound}
                max={status.evaluation_threshold_max}
                step={0.01}
                value={evalThreshold ?? status.confirmed_threshold}
                onChange={(e) => {
                  const v = Math.round(Number(e.target.value) * 100) / 100
                  setEvalThreshold(v === status.confirmed_threshold ? null : v)
                }}
                className="w-full accent-brand"
              />
              <div className="flex justify-between font-mono text-[11px] text-muted-foreground">
                <span>{status.uncertain_lower_bound.toFixed(2)}</span>
                <span>{status.evaluation_threshold_max.toFixed(2)}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Sent to the backend with the next photo you upload and applied by the real matching step. It only moves
                the Confirmed boundary for this browser tab — the Uncertain lower bound stays{" "}
                {status.uncertain_lower_bound.toFixed(2)} and the configured default (
                {status.confirmed_threshold.toFixed(2)}) is not changed.
              </p>
              {evalThreshold !== null && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setEvalThreshold(null)}
                  className="mt-1 justify-self-start"
                >
                  <RotateCcw />
                  Reset to default ({status.confirmed_threshold.toFixed(2)})
                </Button>
              )}
            </div>
          </details>
        )}
      </section>
    </div>
  )
}

// `dot`/`edge` match the box colours the API draws on the photo (green / amber / red).
const tones: Record<
  MatchStatus,
  { icon: typeof Check; title: string; label: string; text: string; dot: string; edge: string }
> = {
  confirmed: {
    icon: Check,
    title: "Identity confirmed",
    label: "Confirmed",
    text: "text-green-700",
    dot: "bg-green-600",
    edge: "border-t-green-600",
  },
  uncertain: {
    icon: TriangleAlert,
    title: "Identity not confirmed",
    label: "Uncertain",
    text: "text-amber-700",
    dot: "bg-amber-600",
    edge: "border-t-amber-600",
  },
  unknown: {
    icon: CircleX,
    title: "No enrolled identity matched",
    label: "Unknown",
    text: "text-red-700",
    dot: "bg-red-600",
    edge: "border-t-red-600",
  },
}

// Bounded frame: the face boxes are drawn into the image by the API, so object-contain scales
// them with it. Tall photos are capped in height and letterboxed, never cropped.
function AnnotatedPhoto({ src }: { src: string }) {
  return (
    <div className="flex items-center justify-center overflow-hidden rounded-md border border-border bg-stone-100">
      {/* eslint-disable-next-line @next/next/no-img-element -- server-rendered data URL */}
      <img
        src={src}
        alt="Uploaded photo with detected faces marked"
        className="block h-auto max-h-[360px] w-full object-contain md:max-h-[440px]"
      />
    </div>
  )
}

function FaceSummary({ faces }: { faces: FaceResult[] }) {
  const count = (s: MatchStatus) => faces.filter((f) => f.status === s).length
  const cells = [
    { label: "Faces detected", value: faces.length, text: "text-foreground" },
    ...(["confirmed", "uncertain", "unknown"] as MatchStatus[]).map((s) => ({
      label: tones[s].label,
      value: count(s),
      text: count(s) > 0 ? tones[s].text : "text-muted-foreground",
    })),
  ]
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {cells.map((c) => (
        <div key={c.label} className="rounded-md border border-border px-3 py-2">
          <p className="text-xs text-muted-foreground">{c.label}</p>
          <p className={cn("text-lg font-semibold tabular-nums", c.text)}>{c.value}</p>
        </div>
      ))}
    </div>
  )
}

function FaceResultRow({
  face,
  showIndex,
  faceImage,
  person,
  lower,
  confirmed,
  onEnroll,
}: {
  face: FaceResult
  showIndex: boolean
  faceImage?: string | null
  person?: Person
  lower?: number
  confirmed: number
  onEnroll: () => void
}) {
  const tone = tones[face.status]
  const Icon = tone.icon
  const heading = (
    <p className={cn("flex items-center gap-1.5 text-sm font-medium", tone.text)}>
      <Icon className="size-4 shrink-0" />
      {showIndex && <span className="font-normal text-muted-foreground">Face {face.index} ·</span>}
      {tone.title}
    </p>
  )
  return (
    <div className="flex flex-col gap-3 py-4 first:pt-0 last:pb-0">
      {faceImage ? (
        <div className="flex items-center gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element -- server-rendered data URL */}
          <img
            src={faceImage}
            alt={`Face ${face.index}`}
            className="size-12 shrink-0 rounded-md border border-border object-cover"
          />
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground">Face {face.index}</p>
            <p className={cn("flex items-start gap-1.5 text-sm font-medium", tone.text)}>
              <Icon className="mt-0.5 size-4 shrink-0" />
              {tone.title}
            </p>
          </div>
        </div>
      ) : (
        heading
      )}

      {face.status !== "unknown" && face.person_name && (
        <div className="flex items-center gap-3">
          <PersonAvatar person={person} name={face.person_name} className="size-10 text-xs" />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">{face.person_name}</p>
            <p className="text-xs text-muted-foreground">
              {face.status === "confirmed" ? "Matched identity" : "Closest candidate — context only, not a confirmed match"}
            </p>
          </div>
        </div>
      )}

      {lower !== undefined ? (
        <SimilarityMeter value={face.similarity} lower={lower} confirmed={confirmed} />
      ) : (
        <p className="text-sm">
          Similarity <span className="font-mono font-semibold">{face.similarity.toFixed(2)}</span>
        </p>
      )}

      {face.status === "uncertain" && (
        <p className="text-xs text-muted-foreground">
          Below the confirmed threshold ({confirmed}) — too close to identify confidently.
        </p>
      )}

      {face.status === "unknown" && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-muted-foreground">Not matched with any enrolled person.</p>
          <Button onClick={onEnroll} variant="outline" size="sm">
            Enroll this person
          </Button>
        </div>
      )}
    </div>
  )
}

function LockedState({
  maxAttempts,
  onTryAgain,
  onEnroll,
}: {
  maxAttempts: number
  onTryAgain: () => void
  onEnroll: () => void
}) {
  return (
    <div className="mt-4">
      <p className="flex items-center gap-1.5 text-sm font-medium text-red-700">
        <CircleX className="size-4" /> Identity could not be confidently verified.
      </p>
      <p className="mt-1 text-sm text-muted-foreground">Maximum attempts reached ({maxAttempts}).</p>
      <div className="mt-5 flex flex-wrap justify-end gap-2 border-t border-border pt-5">
        <Button onClick={onEnroll} variant="outline" size="lg" className="px-4">
          Go to enrollment
        </Button>
        <Button onClick={onTryAgain} size="lg" className="px-4">
          Try again
        </Button>
      </div>
    </div>
  )
}
