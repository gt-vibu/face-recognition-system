const clamp = (v: number) => Math.min(Math.max(v, 0), 1)
const pct = (v: number) => `${clamp(v) * 100}%`

/**
 * Cosine similarity on a 0–1 track with the three decision bands from src/config.py
 * (Unknown < lower ≤ Uncertain < confirmed ≤ Confirmed). Shows similarity only — never "confidence".
 */
export function SimilarityMeter({
  value,
  lower,
  confirmed,
}: {
  value: number
  lower: number
  confirmed: number
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between text-sm">
        <span className="text-muted-foreground">Similarity</span>
        <span className="font-mono font-semibold tabular-nums">{value.toFixed(2)}</span>
      </div>
      <div
        role="meter"
        aria-label="Similarity"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(value.toFixed(2))}
        className="relative h-1.5 rounded-full bg-stone-100"
      >
        <span className="absolute inset-y-0 left-0 rounded-l-full bg-red-200" style={{ width: pct(lower) }} />
        <span
          className="absolute inset-y-0 bg-amber-200"
          style={{ left: pct(lower), width: `${(clamp(confirmed) - clamp(lower)) * 100}%` }}
        />
        <span className="absolute inset-y-0 right-0 rounded-r-full bg-green-200" style={{ left: pct(confirmed) }} />
        <span
          className="absolute top-1/2 h-3.5 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground"
          style={{ left: pct(value) }}
        />
      </div>
      <div className="relative h-4 font-mono text-[11px] text-muted-foreground">
        <span className="absolute left-0">0</span>
        <span className="absolute -translate-x-1/2" style={{ left: pct(lower) }}>
          {lower.toFixed(2)}
        </span>
        <span className="absolute -translate-x-1/2" style={{ left: pct(confirmed) }}>
          {confirmed.toFixed(2)}
        </span>
        <span className="absolute right-0">1</span>
      </div>
    </div>
  )
}
