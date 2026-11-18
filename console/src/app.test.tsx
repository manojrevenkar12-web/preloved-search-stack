import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./app/App";

const demo = JSON.parse(readFileSync(`${process.cwd()}/public/demo.json`, "utf8"));

function mount(response: () => Promise<Response> = async () => new Response(JSON.stringify(demo))) {
  vi.stubGlobal("fetch", vi.fn(response));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
}

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", { value: () => ({ matches: true, addEventListener() {}, removeEventListener() {} }), configurable: true });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("compare designs", () => {
  it("shows a loading state, then both designs' results and what the query was read as", async () => {
    mount(() => new Promise((ok) => setTimeout(() => ok(new Response(JSON.stringify(demo))), 50)));
    expect(await screen.findByRole("status", { name: "Loading the search bench" })).toBeInTheDocument();
    const a = await screen.findByRole("region", { name: "Design A" });
    const b = screen.getByRole("region", { name: "Design B" });
    expect((await within(a).findAllByRole("listitem")).length).toBeGreaterThan(5);
    expect((await within(b).findAllByRole("listitem")).length).toBeGreaterThan(5);
    expect(screen.getByLabelText("What the query was read as")).toHaveTextContent("Chanel");
  });

  it("shows a spelling correction for a recorded typo", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "birkn togo" }));
    expect(await screen.findByText("birkn")).toBeInTheDocument();
  });

  it("says what to do when a query was not recorded", async () => {
    mount();
    fireEvent.change(await screen.findByLabelText(/A buyer types/), { target: { value: "something new" } });
    expect((await screen.findAllByText("This query was not recorded for the demo.")).length).toBe(2);
  });

  it("explains a failure and offers a retry", async () => {
    mount(async () => { throw new TypeError("offline"); });
    expect(await screen.findByRole("alert", {}, { timeout: 4000 })).toHaveTextContent("could not be reached");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});

describe("offline vs online", () => {
  it("ranks the offline metrics and lists every design", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Offline vs online" }));
    expect(await screen.findByText(/Which offline metric orders the designs/)).toBeInTheDocument();
    expect(screen.getAllByRole("row").length).toBe(Object.keys(demo.report.online).length);
    expect(screen.getByText(/A\/A tests called significant/)).toBeInTheDocument();
  });
});
