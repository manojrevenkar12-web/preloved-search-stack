import type { ReactNode } from "react";

export function Loading({ label }: { label: string }) {
  return (
    <div role="status" aria-label={label} className="flex flex-col gap-2 py-6">
      {[0, 1, 2, 3].map((i) => <div key={i} className="h-14 animate-pulse rounded-[var(--radius-tag)] bg-rule/60" />)}
    </div>
  );
}

export function Failure({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div role="alert" className="rounded-[var(--radius-field)] border border-loss/40 bg-loss-soft px-4 py-3 text-sm">
      <p>{message}</p>
      {retry && <button onClick={retry} className="mt-2 font-semibold text-garnet underline underline-offset-4">Try again</button>}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-[var(--radius-field)] border border-dashed border-rule px-4 py-6 text-sm text-muted">
      <p className="font-semibold text-ink">{title}</p>
      {children && <div className="mt-1">{children}</div>}
    </div>
  );
}
