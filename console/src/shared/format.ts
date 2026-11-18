export const pct = (v: number, digits = 0) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;
export const share = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
export const eur = (v: number) => new Intl.NumberFormat("en-GB", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(v);
const CONDITION = ["Never worn", "Very good", "Good", "Fair"];
export const condition = (c: number | null) => (c == null ? "Condition not given" : CONDITION[Math.round(c)] ?? `Grade ${c}`);
