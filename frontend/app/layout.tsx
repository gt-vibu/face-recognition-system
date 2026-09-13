import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"

import "./globals.css"
import { SiteNav } from "@/components/site-nav"
import { cn } from "@/lib/utils"

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" })

const fontMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

export const metadata: Metadata = {
  title: "Face Recognition System",
  description: "Local, offline face enrollment and identification.",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn("antialiased", fontMono.variable, "font-sans", geist.variable)}
      style={{ colorScheme: "light" }}
    >
      {/* Light theme only: no theme provider or dark-mode toggle. */}
      <body className="min-h-svh bg-page text-foreground">
        <SiteNav />
        <main className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-6 sm:py-8">{children}</main>
      </body>
    </html>
  )
}
