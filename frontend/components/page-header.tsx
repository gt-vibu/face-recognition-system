/** Centred page title + short description — same treatment as the Enroll page. */
export function PageHeader({ title, description }: { title: string; description?: React.ReactNode }) {
  return (
    <header className="text-center">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {description && <p className="mt-2 text-sm text-pretty text-muted-foreground">{description}</p>}
    </header>
  )
}
