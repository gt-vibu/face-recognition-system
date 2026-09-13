"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/enroll", label: "Enroll" },
  { href: "/identify", label: "Identify" },
  { href: "/people", label: "People" },
]

export function SiteNav() {
  const pathname = usePathname()
  return (
    // One row (and sticky) from 520px; on narrow phones it wraps and scrolls away so it never covers content.
    <header className="z-40 border-b border-border bg-white/95 backdrop-blur min-[520px]:sticky min-[520px]:top-0">
      <div className="mx-auto flex max-w-4xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5 sm:px-6">
        <Link href="/" className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
          {/* Simple lens mark — no icon library glyph. */}
          <span aria-hidden className="grid size-4 place-items-center rounded-full border-[1.5px] border-foreground">
            <span className="size-1.5 rounded-full bg-foreground" />
          </span>
          Face Recognition
        </Link>

        <nav
          aria-label="Main"
          className="-mx-2 flex w-full gap-0.5 text-sm min-[520px]:mx-0 min-[520px]:ml-auto min-[520px]:w-auto sm:gap-1"
        >
          {links.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href)
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "rounded-md px-2.5 py-1.5 text-muted-foreground transition-colors duration-150 hover:bg-black/[0.04] hover:text-foreground sm:px-3",
                  active && "bg-brand/[0.08] font-medium text-brand hover:bg-brand/[0.08] hover:text-brand"
                )}
              >
                {label}
              </Link>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
