"use client";

import type { UIMessage } from "@ai-sdk/react";
import { motion } from "framer-motion";
import { Streamdown } from "streamdown";

import { SparklesIcon } from "./icons";
import { PreviewAttachment } from "./preview-attachment";
import { cn } from "@/lib/utils";
import { Weather } from "./weather";

const blockStyles =
  "rounded-lg border border-border bg-muted/50 p-3 text-sm";

function formatToolName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export const PreviewMessage = ({
  message,
}: {
  chatId: string;
  message: UIMessage;
  isLoading: boolean;
}) => {
  return (
    <motion.div
      className="w-full mx-auto max-w-3xl px-4 group/message"
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      data-role={message.role}
    >
      <div
        className={cn(
          "group-data-[role=user]/message:bg-primary group-data-[role=user]/message:text-primary-foreground flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl"
        )}
      >
        {message.role === "assistant" && (
          <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border">
            <SparklesIcon size={14} />
          </div>
        )}

        <div className="flex flex-col gap-3 w-full">
          {message.parts &&
            message.parts.map((part: any, index: number) => {
              // Stable key per segment so multiple text/reasoning blocks don't get merged (e.g. text before tools vs after tools)
              const partKey =
                part.type?.startsWith("tool-")
                  ? part.toolCallId
                  : part.id ?? `${part.type ?? "part"}-${index}`;

              if (part.type === "text") {
                return (
                  <div key={partKey} className="flex flex-col gap-4">
                    <Streamdown>{part.text}</Streamdown>
                  </div>
                );
              }
              if (part.type === "reasoning" && part.text) {
                return (
                  <div
                    key={partKey}
                    className={cn(blockStyles, "text-muted-foreground")}
                  >
                    <div className="font-medium text-muted-foreground/80 mb-2">
                      Reasoning
                    </div>
                    <div className="whitespace-pre-wrap">
                      <Streamdown>{part.text}</Streamdown>
                    </div>
                  </div>
                );
              }
              if (part.type?.startsWith("tool-")) {
                const { toolCallId, state, output, input } = part;
                const toolName = part.type.replace("tool-", "");
                const displayName = formatToolName(toolName);
                const args = input ?? part.args;
                const hasOutput = state === "output-available" && output != null;
                const hasOutputError =
                  state === "output-error" && part.errorText;
                const isWeather = toolName === "get_current_weather";

                return (
                  <div
                    key={partKey}
                    className={cn(blockStyles, {
                      "border-destructive/50 bg-destructive/5":
                        hasOutputError,
                    })}
                  >
                    <div className="font-medium text-muted-foreground/80 mb-2">
                      Tool: {displayName}
                    </div>
                    <div className="flex flex-col gap-3">
                      <div>
                        <span className="text-muted-foreground/80 font-medium text-xs uppercase tracking-wide">
                          Input
                        </span>
                        <div className="mt-1">
                          {args != null &&
                          (typeof args === "string"
                            ? args.length > 0
                            : Object.keys(args).length > 0) ? (
                            <pre className="overflow-x-auto text-xs">
                              {typeof args === "string"
                                ? args
                                : JSON.stringify(args, null, 2)}
                            </pre>
                          ) : (
                            <span className="text-muted-foreground italic text-xs">
                              Calling…
                            </span>
                          )}
                        </div>
                      </div>
                      <div>
                        <span className="text-muted-foreground/80 font-medium text-xs uppercase tracking-wide">
                          Output
                        </span>
                        <div className="mt-1">
                          {hasOutputError ? (
                            <div className="text-destructive text-xs">
                              {part.errorText}
                            </div>
                          ) : hasOutput ? (
                            isWeather ? (
                              <Weather weatherAtLocation={output} />
                            ) : (
                              <pre className="overflow-x-auto text-xs">
                                {JSON.stringify(output, null, 2)}
                              </pre>
                            )
                          ) : isWeather && (state === "input-streaming" || state === "input-available") ? (
                            <Weather />
                          ) : (
                            <span className="text-muted-foreground italic text-xs">
                              …
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              }
              if (part.type === "file") {
                return (
                  <PreviewAttachment
                    key={partKey}
                    attachment={part}
                  />
                );
              }
              return null;
            })}
        </div>
      </div>
    </motion.div>
  );
};

export const ThinkingMessage = () => {
  const role = "assistant";

  return (
    <motion.div
      className="w-full mx-auto max-w-3xl px-4 group/message "
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1, transition: { delay: 1 } }}
      data-role={role}
    >
      <div
        className={cn(
          "flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl",
          {
            "group-data-[role=user]/message:bg-muted": true,
          }
        )}
      >
        <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border">
          <SparklesIcon size={14} />
        </div>

        <div className="flex flex-col gap-2 w-full">
          <div className="flex flex-col gap-4 text-muted-foreground">
            Thinking...
          </div>
        </div>
      </div>
    </motion.div>
  );
};
