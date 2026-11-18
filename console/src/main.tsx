import "@fontsource/bodoni-moda/400.css";
import "@fontsource/bodoni-moda/600.css";
import "@fontsource/hanken-grotesk/400.css";
import "@fontsource/hanken-grotesk/600.css";
import "./theme/tokens.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app/App";

const queries = new QueryClient();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queries}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
