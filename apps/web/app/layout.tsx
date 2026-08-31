import "./globals.css";

export const metadata = {
  title: "AgentForge",
  description: "Playground, traces, evals, and tool catalog for the AgentForge runtime",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="dark">
      <body>{children}</body>
    </html>
  );
}
