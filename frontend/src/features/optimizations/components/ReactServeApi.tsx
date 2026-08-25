"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { Separator } from "@/shared/ui/primitives/separator";
import { Skeleton } from "@/shared/ui/skeleton";
import { HelpTip } from "@/shared/ui/help-tip";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { msg } from "@/shared/lib/messages";
import { tip } from "@/shared/lib/tooltips";

import { CopyButton, LangPicker } from "./ui-primitives";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={180} borderRadius={8} />,
});

export interface ReactServeApiProps {
  optimizationId: string;
}

// Owner-facing "serving API" panel for an optimized ReAct agent: the live chat
// endpoint plus copy-paste integration snippets. The transport differs from the
// generic /serve endpoint (a streaming SSE chat turn, not a single inference),
// so ServeCodeSnippets can't be reused — the request/response shapes don't overlap.
export function ReactServeApi({ optimizationId }: ReactServeApiProps) {
  const [lang, setLang] = useState<"curl" | "python" | "javascript" | "go" | "dspy">("curl");
  const apiBase = getRuntimeEnv().apiUrl;
  const url = `${apiBase}/serve/${optimizationId}/chat`;
  const confirmPath = `/serve/${optimizationId}/chat/confirm`;
  const artifactUrl = `${apiBase}/optimizations/${optimizationId}/artifact`;

  const snippets = {
    curl: [
      `# Generate a token in Settings → API, then set it in your environment`,
      `#   export SKYNET_API_TOKEN=skyd_...`,
      `# Stream one chat turn against the optimized ReAct agent (Server-Sent Events).`,
      `# trust_mode "yolo" runs tools without approval; "ask" emits pending_approval`,
      `# events you resolve via POST ${confirmPath}.`,
      `curl -N -X POST ${url} \\`,
      `  -H "Authorization: Bearer $SKYNET_API_TOKEN" \\`,
      `  -H "Content-Type: application/json" \\`,
      `  -d '{"user_message": "<your message>", "trust_mode": "yolo"}'`,
    ].join("\n"),
    python: [
      `import os`,
      `import requests`,
      ``,
      `# Generate a token in Settings → API and set SKYNET_API_TOKEN in your env`,
      `token = os.environ["SKYNET_API_TOKEN"]`,
      ``,
      `# Stream one chat turn against the optimized ReAct agent (Server-Sent Events).`,
      `# trust_mode "yolo" runs tools without approval; "ask" pauses for confirmation.`,
      `with requests.post(`,
      `    "${url}",`,
      `    headers={"Authorization": f"Bearer {token}"},`,
      `    json={"user_message": "<your message>", "trust_mode": "yolo"},`,
      `    stream=True,`,
      `) as response:`,
      `    for line in response.iter_lines():`,
      `        if line:`,
      `            print(line.decode())`,
    ].join("\n"),
    javascript: [
      `// Generate a token in Settings → API and set SKYNET_API_TOKEN in your env`,
      `const token = process.env.SKYNET_API_TOKEN;`,
      ``,
      `// Stream one chat turn against the optimized ReAct agent (Server-Sent Events).`,
      `// trust_mode "yolo" runs tools without approval; "ask" pauses for confirmation.`,
      `const response = await fetch("${url}", {`,
      `  method: "POST",`,
      `  headers: {`,
      `    "Authorization": "Bearer " + token,`,
      `    "Content-Type": "application/json",`,
      `  },`,
      `  body: JSON.stringify({ user_message: "<your message>", trust_mode: "yolo" }),`,
      `});`,
      ``,
      `// Read the Server-Sent Events stream chunk by chunk`,
      `const reader = response.body.getReader();`,
      `const decoder = new TextDecoder();`,
      `for (;;) {`,
      `  const { value, done } = await reader.read();`,
      `  if (done) break;`,
      `  process.stdout.write(decoder.decode(value));`,
      `}`,
    ].join("\n"),
    go: [
      `package main`,
      ``,
      `import (`,
      `\t"bufio"`,
      `\t"bytes"`,
      `\t"encoding/json"`,
      `\t"fmt"`,
      `\t"net/http"`,
      `\t"os"`,
      `)`,
      ``,
      `func main() {`,
      `\t// Generate a token in Settings → API and set SKYNET_API_TOKEN in your env`,
      `\ttoken := os.Getenv("SKYNET_API_TOKEN")`,
      ``,
      `\t// Stream one chat turn against the optimized ReAct agent (Server-Sent Events).`,
      `\t// trust_mode "yolo" runs tools without approval; "ask" pauses for confirmation.`,
      `\tpayload, _ := json.Marshal(map[string]any{`,
      `\t\t"user_message": "<your message>",`,
      `\t\t"trust_mode":   "yolo",`,
      `\t})`,
      `\treq, _ := http.NewRequest("POST", "${url}", bytes.NewReader(payload))`,
      `\treq.Header.Set("Authorization", "Bearer "+token)`,
      `\treq.Header.Set("Content-Type", "application/json")`,
      `\tresp, _ := http.DefaultClient.Do(req)`,
      `\tdefer resp.Body.Close()`,
      ``,
      `\t// Read the Server-Sent Events stream line by line`,
      `\tscanner := bufio.NewScanner(resp.Body)`,
      `\tfor scanner.Scan() {`,
      `\t\tfmt.Println(scanner.Text())`,
      `\t}`,
      `}`,
    ].join("\n"),
    dspy: [
      `import os`,
      `import asyncio`,
      `import requests`,
      `import dspy`,
      `from mcp import ClientSession`,
      `from mcp.client.streamable_http import streamablehttp_client`,
      ``,
      `# Generate a token in Settings → API and set SKYNET_API_TOKEN in your env`,
      `token = os.environ["SKYNET_API_TOKEN"]`,
      `auth = {"Authorization": f"Bearer {token}"}`,
      ``,
      `# Define the DSPy Signature you trained this agent with (the one you submitted)`,
      `class Agent(dspy.Signature):`,
      `    user_message: str = dspy.InputField()`,
      `    assistant_message: str = dspy.OutputField()`,
      ``,
      `# Download the optimized ReAct artifact (state + tool overlay + MCP source)`,
      `artifact = requests.get("${artifactUrl}", headers=auth).json()["program_artifact"]`,
      `overlay = artifact["react_overlay"]`,
      `source = overlay["tool_source"]`,
      ``,
      ``,
      `async def main():`,
      `    # ReAct tools bind to a live MCP session, so rebuild the roster from your`,
      `    # MCP server rather than unpickling a dead-session program.`,
      `    async with (`,
      `        streamablehttp_client(source["mcp_url"], headers=auth) as (read, write, _),`,
      `        ClientSession(read, write) as session,`,
      `    ):`,
      `        await session.initialize()`,
      `        listing = await session.list_tools()`,
      `        tools = [dspy.Tool.from_mcp_tool(session, t) for t in listing.tools]`,
      `        if source.get("tool_filter"):`,
      `            by_name = {t.name: t for t in tools}`,
      `            tools = [by_name[n] for n in source["tool_filter"] if n in by_name]`,
      ``,
      `        # Re-apply GEPA's optimized tool wording + renames so the live roster`,
      `        # matches the names baked into the loaded instructions.`,
      `        for t in tools:`,
      `            if overlay["tool_descriptions"].get(t.name):`,
      `                t.desc = overlay["tool_descriptions"][t.name]`,
      `        for t in tools:`,
      `            renamed = (overlay.get("tool_names") or {}).get(t.name)`,
      `            if renamed:`,
      `                t.name = renamed`,
      ``,
      `        program = dspy.ReActV2(Agent, tools=tools, max_iters=overlay["max_iters"])`,
      `        program.load_state(artifact["program_state_json"])`,
      ``,
      `        with dspy.context(lm=dspy.LM("<your model>")):`,
      `            result = program(user_message="<your message>")`,
      `        for name in Agent.output_fields:`,
      `            print(getattr(result, name))`,
      ``,
      ``,
      `asyncio.run(main())`,
    ].join("\n"),
  };
  const labels = {
    curl: "cURL",
    python: "Python",
    javascript: "JavaScript",
    go: "Go",
    dspy: "DSPy",
  } as const;
  const snippet = snippets[lang];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">
          <HelpTip text={tip("serve.section_run")}>{msg("optimizations.react.api_title")}</HelpTip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <p className="text-[0.625rem] text-muted-foreground uppercase tracking-wider">
            <HelpTip text={tip("serve.api_url_react")}>
              {msg("auto.app.optimizations.id.page.23")}
            </HelpTip>
          </p>
          <div className="group relative rounded-lg bg-muted/40 p-2.5 pe-11 sm:pe-8" dir="ltr">
            <code className="text-xs font-mono break-all">{url}</code>
            <CopyButton
              text={url}
              className="absolute end-1 top-1 opacity-100 sm:end-1.5 sm:top-1.5 sm:opacity-0 sm:group-hover:opacity-100"
            />
          </div>
        </div>

        <Separator />

        <div className="space-y-2">
          <p className="text-[0.625rem] text-muted-foreground uppercase tracking-wider">
            <HelpTip text={tip("serve.integration_code")}>
              {msg("auto.app.optimizations.id.page.26")}
            </HelpTip>
          </p>
          <CodeEditor
            value={snippet}
            onChange={() => {}}
            height={`${(snippet.split("\n").length + 1) * 19.6 + 8}px`}
            readOnly
            label={<LangPicker value={lang} onChange={setLang} labels={labels} />}
          />
        </div>
      </CardContent>
    </Card>
  );
}
