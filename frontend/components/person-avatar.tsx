import { thumbnailUrl, type Person } from "@/lib/api"
import { cn } from "@/lib/utils"

/** Enrolled person's thumbnail, falling back to initials. */
export function PersonAvatar({
  person,
  name,
  className,
}: {
  person?: Person
  name: string
  className?: string
}) {
  if (person?.has_thumbnail) {
    return (
      // eslint-disable-next-line @next/next/no-img-element -- served by the local Python API
      <img
        src={thumbnailUrl(person)}
        alt={name}
        className={cn("shrink-0 rounded-md object-cover ring-1 ring-black/5", className)}
      />
    )
  }
  const initials = name
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase()
  return (
    <span
      aria-hidden
      className={cn(
        "flex shrink-0 items-center justify-center rounded-md bg-stone-100 font-semibold text-stone-600",
        className
      )}
    >
      {initials}
    </span>
  )
}
