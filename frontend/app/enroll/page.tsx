"use client"

import * as React from "react"
import Link from "next/link"
import { ArrowRight, Check, CircleX, LoaderCircle, Plus, TriangleAlert, X } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button, buttonVariants } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useSystemStatus } from "@/hooks/use-system-status"
import { api, errorMessage, type Person } from "@/lib/api"
import { cn } from "@/lib/utils"

const ACCEPTED = ["image/jpeg", "image/png"]

type PhotoCheck = "checking" | "ok" | "none" | "multiple" | "error"

type Photo = {
  id: string
  file: File
  previewUrl: string
  check: PhotoCheck
}

export default function EnrollPage() {
  const { status, error: statusError } = useSystemStatus()
  const [name, setName] = React.useState("")
  const [photos, setPhotos] = React.useState<Photo[]>([])
  const [ignored, setIgnored] = React.useState(0)
  const [wrongType, setWrongType] = React.useState(false)
  const [dragging, setDragging] = React.useState(false)
  const dragDepth = React.useRef(0)
  const [existing, setExisting] = React.useState<Person[]>([])
  // For an already-enrolled name the user must pick what saving does; reset whenever the name changes.
  const [choice, setChoice] = React.useState<"add" | "replace" | null>(null)
  const [saving, setSaving] = React.useState(false)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const [saved, setSaved] = React.useState<string | null>(null)

  // Limits come from src/config.py via the API (2–5 by default).
  const minImages = status?.min_enrollment_images ?? 2
  const maxImages = status?.max_enrollment_images ?? 5

  React.useEffect(() => {
    api.persons().then(setExisting, () => {})
  }, [saved])

  // "Add photos" on the People page links here as /enroll?name=… — prefill that name.
  React.useEffect(() => {
    const preset = new URLSearchParams(window.location.search).get("name")
    if (preset) setName(preset)
  }, [])

  // Preview object URLs are freed when a photo is removed, after saving, or on unmount.
  const previewUrls = React.useRef(new Map<string, string>())
  React.useEffect(() => {
    const urls = previewUrls.current
    return () => urls.forEach((url) => URL.revokeObjectURL(url))
  }, [])

  function addPhotos(incoming: File[]) {
    const images = incoming.filter((f) => ACCEPTED.includes(f.type))
    setWrongType(images.length < incoming.length)
    setSaved(null)
    setSaveError(null)
    const room = Math.max(maxImages - photos.length, 0)
    setIgnored(Math.max(images.length - room, 0))
    const added = images.slice(0, room).map((file) => {
      const photo: Photo = { id: crypto.randomUUID(), file, previewUrl: URL.createObjectURL(file), check: "checking" }
      previewUrls.current.set(photo.id, photo.previewUrl)
      return photo
    })
    setPhotos((prev) => [...prev, ...added])

    for (const photo of added) {
      api
        .detect(photo.file)
        .then((r): PhotoCheck => (r.face_count === 0 ? "none" : r.face_count > 1 ? "multiple" : "ok"))
        .catch((): PhotoCheck => "error")
        .then((check) => setPhotos((prev) => prev.map((p) => (p.id === photo.id ? { ...p, check } : p))))
    }
  }

  function removePhoto(id: string) {
    const url = previewUrls.current.get(id)
    if (url) URL.revokeObjectURL(url)
    previewUrls.current.delete(id)
    setIgnored(0)
    setPhotos((prev) => prev.filter((p) => p.id !== id))
  }

  function clearPhotos() {
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url))
    previewUrls.current.clear()
    setPhotos([])
    setIgnored(0)
    setWrongType(false)
  }

  const nameClean = name.trim()
  const usable = photos.filter((p) => p.check === "ok")
  const unusableCount = photos.filter((p) => p.check !== "ok" && p.check !== "checking").length
  const stillChecking = photos.some((p) => p.check === "checking")
  const existingPerson = nameClean === "" ? undefined : existing.find((p) => p.name === nameClean)
  // "new" for a new name; for an existing one, "add" by default when possible — never an implicit replace.
  const mode: "new" | "add" | "replace" | null = !existingPerson
    ? "new"
    : (choice ?? (existingPerson.can_add_photos ? "add" : null))
  // Adding to an existing enrollment needs only one photo; new and replace keep the usual minimum.
  const requiredPhotos = mode === "add" ? 1 : minImages
  const ready = nameClean !== "" && mode !== null && usable.length >= requiredPhotos && !stillChecking && !saving
  const full = photos.length >= maxImages
  const enough = usable.length >= requiredPhotos && !stillChecking
  // Show the required slots plus one more, growing as photos are added (never more than the max).
  const visibleSlots = Math.min(maxImages, Math.max(requiredPhotos + 1, photos.length + 1))

  async function save() {
    setSaving(true)
    setSaveError(null)
    try {
      const files = usable.map((p) => p.file)
      if (mode === "add" && existingPerson) {
        const res = await api.addPhotos(existingPerson.person_id, files)
        setSaved(`Added ${res.added} photo${res.added === 1 ? "" : "s"} to ${res.name} — ${res.num_samples} in total.`)
      } else {
        const res = await api.enroll(nameClean, files, mode === "replace" ? "replace" : "new")
        setSaved(`${res.name} ${res.mode === "replace" ? "re-enrolled" : "enrolled"} with ${res.num_samples} photos.`)
      }
      setName("")
      setChoice(null)
      clearPhotos()
    } catch (e) {
      setSaveError(errorMessage(e))
    } finally {
      setSaving(false)
    }
  }

  function progressText() {
    if (stillChecking) return "Checking photos…"
    if (usable.length < requiredPhotos) {
      const need = requiredPhotos - usable.length
      return `${usable.length} of ${requiredPhotos} photos added — add at least ${need === 1 ? "one more" : `${need} more`}`
    }
    return `${usable.length} photo${usable.length === 1 ? "" : "s"} ready`
  }

  return (
    <div className="mx-auto flex max-w-xl flex-col gap-6 sm:pt-2">
      <header className="text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Enroll a person</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          {mode === "add" && existingPerson
            ? `Add 1–${maxImages} more photos of ${existingPerson.name}.`
            : `Add ${minImages}–${maxImages} photos of the same person.`}
          <br className="hidden sm:block" /> For best results, add 3 photos with slightly different angles or lighting.
        </p>
      </header>

      {statusError && (
        <Alert variant="destructive" className="border-red-200 bg-red-50">
          <CircleX />
          <AlertTitle>Recognition API not reachable</AlertTitle>
          <AlertDescription>{statusError}</AlertDescription>
        </Alert>
      )}

      {saved && (
        <p className="flex items-center justify-between gap-3 rounded-md border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-800">
          <span className="flex items-center gap-1.5">
            <Check className="size-4" /> {saved}
          </span>
          <Link href="/identify" className="shrink-0 font-medium underline-offset-4 hover:underline">
            Identify a face
          </Link>
        </p>
      )}

      <section
        aria-label="Enrollment"
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
          if (!full) addPhotos(Array.from(e.dataTransfer.files))
        }}
      >
        {/* Photos */}
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-medium">Photos</h2>
          <span className="text-sm text-muted-foreground tabular-nums">
            {photos.length} / {maxImages}
          </span>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
          <label
            htmlFor="photos"
            className={cn(
              buttonVariants({ variant: "outline", size: "sm" }),
              "cursor-pointer",
              full && "pointer-events-none opacity-50"
            )}
          >
            <Plus /> Add photos
          </label>
          <span className="text-xs text-muted-foreground">
            {dragging ? "Drop to add" : "JPG or PNG · One face per photo"}
          </span>
        </div>
        <input
          id="photos"
          type="file"
          accept={ACCEPTED.join(",")}
          multiple
          disabled={full}
          className="sr-only"
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            e.target.value = "" // allow re-selecting the same file
            if (files.length) addPhotos(files)
          }}
        />

        <ul className="mt-4 grid grid-cols-2 gap-3 min-[480px]:grid-cols-3">
          {photos.map((photo) => (
            <PhotoCard key={photo.id} photo={photo} onRemove={() => removePhoto(photo.id)} />
          ))}
          {Array.from({ length: Math.max(visibleSlots - photos.length, 0) }, (_, i) => (
            <AddSlot key={`slot-${photos.length + i}`} optional={photos.length + i >= requiredPhotos} />
          ))}
        </ul>

        {(photos.length > 0 || ignored > 0 || wrongType) && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-sm">
            <p className={cn("flex flex-wrap items-center gap-x-1.5", enough ? "text-foreground" : "text-muted-foreground")}>
              {photos.length > 0 && (
                <>
                  {enough && <Check className="size-4 text-green-600" />}
                  {progressText()}
                </>
              )}
              {unusableCount > 0 && !stillChecking && (
                <span className="text-red-700">
                  · {unusableCount} can&apos;t be used — remove {unusableCount > 1 ? "them" : "it"}
                </span>
              )}
              {ignored > 0 && <span className="text-amber-700">· {ignored} extra ignored (max {maxImages})</span>}
              {wrongType && <span className="text-red-700">· Only JPG or PNG images are supported</span>}
            </p>
            {photos.length > 0 && (
              <button
                type="button"
                onClick={clearPhotos}
                className="text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
              >
                Clear all
              </button>
            )}
          </div>
        )}

        {/* Name + save, part of the same workflow */}
        <div className="mt-6 grid gap-2">
          <Label htmlFor="name">Name</Label>
          <Input
            id="name"
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              setChoice(null)
            }}
            placeholder="e.g. Priya Sharma"
            className="h-9"
          />
          {!existingPerson && (
            <p className="text-xs text-muted-foreground">Already enrolled? Enter their exact name to add more photos.</p>
          )}
          {existingPerson && (
            <ExistingPersonChoice
              person={existingPerson}
              mode={mode}
              minImages={minImages}
              onChoose={setChoice}
            />
          )}
          {saveError && <p className="text-xs text-red-700">{saveError}</p>}
        </div>

        <div className="mt-5 flex justify-end">
          <Button onClick={save} disabled={!ready} size="lg" className="px-4">
            {saving ? <LoaderCircle className="animate-spin" /> : null}
            {mode === "add" ? "Add photos" : mode === "replace" ? "Replace enrollment" : "Save person"}
            {!saving && <ArrowRight />}
          </Button>
        </div>
      </section>
    </div>
  )
}

// Shown when the typed name is already enrolled: saving must be an explicit "add" or "replace".
function ExistingPersonChoice({
  person,
  mode,
  minImages,
  onChoose,
}: {
  person: Person
  mode: "new" | "add" | "replace" | null
  minImages: number
  onChoose: (choice: "add" | "replace") => void
}) {
  const option = (value: "add" | "replace", label: string, disabled = false) => (
    <button
      type="button"
      role="radio"
      aria-checked={mode === value}
      disabled={disabled}
      onClick={() => onChoose(value)}
      className={cn(
        buttonVariants({ variant: mode === value ? "default" : "outline", size: "sm" }),
        "disabled:opacity-40"
      )}
    >
      {label}
    </button>
  )
  return (
    <div className="mt-1 grid gap-2 rounded-md border border-border bg-stone-50 p-3">
      <p className="flex items-center gap-1.5 text-xs text-amber-700">
        <TriangleAlert className="size-3.5 shrink-0" />
        {person.name} is already enrolled ({person.num_samples} photo{person.num_samples === 1 ? "" : "s"}). Choose
        what saving does:
      </p>
      <div role="radiogroup" aria-label="Existing enrollment" className="flex flex-wrap gap-2">
        {option("add", "Add photos", !person.can_add_photos)}
        {option("replace", "Replace enrollment")}
      </div>
      <p className="text-xs text-muted-foreground">
        {!person.can_add_photos && mode !== "replace"
          ? "Adding photos isn't available for this person (enrolled before it was supported). Replace their enrollment once to enable it."
          : mode === "add"
            ? "Adds these photos (1 or more) to their existing enrollment. Only face embeddings are stored — never the photos."
            : `Discards their current reference and re-enrolls from these photos only (at least ${minImages}).`}
      </p>
    </div>
  )
}

const checkText: Record<PhotoCheck, string> = {
  checking: "Checking…",
  ok: "Face found",
  none: "No face found",
  multiple: "Multiple faces",
  error: "Unreadable",
}

function PhotoCard({ photo, onRemove }: { photo: Photo; onRemove: () => void }) {
  const bad = photo.check === "none" || photo.check === "multiple" || photo.check === "error"
  return (
    <li className="relative overflow-hidden rounded-md border border-border bg-white">
      {/* eslint-disable-next-line @next/next/no-img-element -- local object URL preview */}
      <img src={photo.previewUrl} alt={photo.file.name} className="aspect-square w-full bg-stone-100 object-cover" />
      <p
        className={cn(
          "flex items-center gap-1.5 px-2.5 py-2 text-xs",
          bad ? "text-red-700" : photo.check === "ok" ? "text-foreground" : "text-muted-foreground"
        )}
      >
        {photo.check === "checking" && <LoaderCircle className="size-3.5 animate-spin" />}
        {photo.check === "ok" && <Check className="size-3.5 text-green-600" />}
        {bad && <CircleX className="size-3.5" />}
        {checkText[photo.check]}
      </p>
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${photo.file.name}`}
        title="Remove"
        className="absolute top-1.5 right-1.5 flex size-6 items-center justify-center rounded-md bg-white/90 text-foreground shadow-sm transition-colors hover:bg-white focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none"
      >
        <X className="size-3.5" />
      </button>
    </li>
  )
}

function AddSlot({ optional }: { optional: boolean }) {
  return (
    <li className="h-full">
      <label
        htmlFor="photos"
        className="flex h-full min-h-32 cursor-pointer flex-col items-center justify-center gap-1 rounded-md border border-dashed border-stone-300 text-muted-foreground transition-colors hover:border-stone-400 hover:bg-stone-50 hover:text-foreground"
      >
        <Plus className="size-4" />
        <span className="text-sm">Add photo</span>
        {optional && <span className="text-xs text-muted-foreground/80">Optional</span>}
      </label>
    </li>
  )
}
