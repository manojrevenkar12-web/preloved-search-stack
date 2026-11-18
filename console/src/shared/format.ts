export const pct = (v: number, digits = 0) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;
export const share = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
export const eur = (v: number) => new Intl.NumberFormat("en-GB", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(v);
/** The seller's condition grade, as listed ("very good"), sentence-cased. */
export const condition = (c: string | null) => (c ? c[0].toUpperCase() + c.slice(1) : "Condition not given");
