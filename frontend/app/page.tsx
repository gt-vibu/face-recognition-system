"use client"

import Link from "next/link"
import { ArrowRight, CircleX } from "lucide-react"

import { PageHeader } from "@/components/page-header"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { useSystemStatus } from "@/hooks/use-system-status"
import { cn } from "@/lib/utils"

export default function DashboardPage() {
  const { status, error } = useSystemStatus()

  return (
    <div className="mx-auto flex max-w-xl flex-col gap-6 sm:pt-2">
      <PageHeader
        title="Identify faces locally"
        description="Enroll people from a few photos, then identify them in new images using ArcFace embeddings and cosine similarity."
      />

      {error && (
        <Alert variant="destructive" className="border-red-200 bg-red-50">
          <CircleX />
          <AlertTitle>Recognition API not reachable</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <section className="rounded-lg border border-border bg-white p-5 shadow-[0_1px_2px_rgb(0_0_0/0.03)] sm:p-6">
        {/* Workflow */}
        <h2 className="text-sm font-medium">Workflow</h2>
        <ul className="mt-3 divide-y divide-border">
          <li className="flex items-center justify-between gap-4 pb-4">
            <div>
              <p className="text-sm font-medium">Identify a face</p>
              <p className="mt-0.5 text-xs text-muted-foreground">Match a new photo against everyone enrolled.</p>
            </div>
            <Button nativeButton={false} render={<Link href="/identify" />} size="lg" className="shrink-0 px-4">
              Identify <ArrowRight />
            </Button>
          </li>
          <li className="flex items-center justify-between gap-4 pt-4">
            <div>
              <p className="text-sm font-medium">Enroll a person</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Add {status ? `${status.min_enrollment_images}–${status.max_enrollment_images}` : "2–5"} photos to create
                their reference.
              </p>
            </div>
            <Button nativeButton={false} render={<Link href="/enroll" />} variant="outline" size="sm" className="shrink-0">
              Enroll
            </Button>
          </li>
        </ul>

        {/* System */}
        <div className="mt-6 flex items-baseline justify-between border-t border-border pt-5">
          <h2 className="text-sm font-medium">System</h2>
          <span
            className={cn(
              "flex items-center gap-1.5 text-sm",
              error ? "text-red-700" : status ? "text-green-700" : "text-muted-foreground"
            )}
          >
            <span
              aria-hidden
              className={cn("size-1.5 rounded-full", error ? "bg-red-600" : status ? "bg-green-600" : "bg-stone-300")}
            />
            {error ? "Offline" : status ? "Ready" : "Connecting…"}
          </span>
        </div>
        <dl className="mt-3 grid grid-cols-3 gap-3">
          <Figure label="Enrolled people">
            {status ? (
              <Link href="/people" className="underline-offset-4 hover:underline">
                {status.enrolled_count}
              </Link>
            ) : (
              "—"
            )}
          </Figure>
          <Figure label="Match threshold">{status ? status.confirmed_threshold.toFixed(2) : "—"}</Figure>
          <Figure label="Retry limit">{status ? `${status.max_identification_attempts} attempts` : "—"}</Figure>
        </dl>
        <p className="mt-4 text-xs text-muted-foreground">SCRFD · ArcFace · ONNX Runtime · CPU · Local processing</p>
      </section>
    </div>
  )
}

function Figure({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col-reverse gap-0.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-lg font-semibold tabular-nums">{children}</dd>
    </div>
  )
}
