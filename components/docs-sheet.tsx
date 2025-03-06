"use client";

import { useEffect, useState } from "react";
import { Sheet, SheetContent, SheetTrigger, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import Spinner from "@/components/spinner";
import { CodeIcon, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeSection {
    title: string;
    description: string;
    code: string;
}

function parseCodeSections(code: string): CodeSection[] {
    const lines = code.split('\n');
    const sections: CodeSection[] = [];
    let currentSection: CodeSection | null = null;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line.startsWith('# region')) {
            currentSection = {
                title: line.substring(8),
                description: '',
                code: ''
            };

            // Look for docstring
            while (i + 1 < lines.length) {
                i++;
                const nextLine = lines[i].trim();
                if (nextLine.startsWith('"""')) {
                    i++;
                    while (i < lines.length && !lines[i].trim().endsWith('"""')) {
                        currentSection.description += lines[i] + '\n';
                        i++;
                    }
                    currentSection.description = currentSection.description.trim();
                    break;
                }
                if (nextLine && !nextLine.startsWith('"""')) {
                    i--;
                    break;
                }
            }
            continue;
        }

        if (line === '# endregion' && currentSection) {
            sections.push(currentSection);
            currentSection = null;
            continue;
        }

        if (currentSection) {
            currentSection.code += lines[i] + '\n';
        }
    }

    return sections;
}

export default function DocsSheet() {
    const [sections, setSections] = useState<CodeSection[]>([]);
    const [loading, setLoading] = useState(true);
    const [expandedSections, setExpandedSections] = useState<boolean[]>([]);

    useEffect(() => {
        console.log("Source URL:", process.env.NEXT_PUBLIC_GT_CLOUD_STRUCTURE_SOURCE_URL);

        if (!process.env.NEXT_PUBLIC_GT_CLOUD_STRUCTURE_SOURCE_URL) {
            console.error("Missing NEXT_PUBLIC_GT_CLOUD_STRUCTURE_SOURCE_URL environment variable");
            setLoading(false);
            return;
        }

        fetch(process.env.NEXT_PUBLIC_GT_CLOUD_STRUCTURE_SOURCE_URL)
            .then((res) => {
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                return res.text();
            })
            .then((text) => {
                console.log("Fetched content length:", text.length); // Debug log
                const parsedSections = parseCodeSections(text);
                setSections(parsedSections);
                setExpandedSections(parsedSections.map(section =>
                    section.title === "Data Model" || section.title === "Agent Configuration"
                ));
                setLoading(false);
            })
            .catch((err) => {
                console.error("Failed to load docs:", err);
                setLoading(false);
            });
    }, []);

    const toggleSection = (index: number) => {
        setExpandedSections(prev => {
            const next = [...prev];
            next[index] = !next[index];
            return next;
        });
    };

    return (
        <Sheet>
            <SheetTrigger asChild>
                <Button size="icon" variant="outline">
                    <CodeIcon className="h-4 w-4" />
                </Button>
            </SheetTrigger>
            <SheetContent>
                <SheetTitle>Griptape Agent Code</SheetTitle>
                <SheetDescription>
                    Documentation showing the Griptape agent code structure and implementation
                </SheetDescription>
                <div className="flex flex-col h-[calc(100%-3rem)] overflow-y-auto mt-4">
                    {loading ? (
                        <div className="flex items-center justify-center flex-1">
                            <Spinner />
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {sections.map((section, index) => (
                                <div key={index} className="border-b border-border pb-6 last:border-0">
                                    <div
                                        className="flex items-center justify-between cursor-pointer mb-2"
                                        onClick={() => toggleSection(index)}
                                    >
                                        <h3 className="text-lg font-semibold">{section.title}</h3>
                                        {expandedSections[index] ?
                                            <ChevronUp className="h-4 w-4" /> :
                                            <ChevronDown className="h-4 w-4" />
                                        }
                                    </div>
                                    {expandedSections[index] && (
                                        <>
                                            {section.description && (
                                                <p className="text-sm text-muted-foreground mb-4">
                                                    {section.description}
                                                </p>
                                            )}
                                            <SyntaxHighlighter
                                                language="python"
                                                style={{
                                                    ...oneDark,
                                                    'pre[class*="language-"]': {
                                                        ...oneDark['pre[class*="language-"]'],
                                                        background: 'transparent',
                                                        margin: 0,
                                                        padding: '1rem',
                                                    },
                                                    'code[class*="language-"]': {
                                                        ...oneDark['code[class*="language-"]'],
                                                        background: 'transparent',
                                                    },
                                                }}
                                                showLineNumbers
                                                customStyle={{
                                                    fontSize: '12px',
                                                }}
                                            >
                                                {section.code.trim()}
                                            </SyntaxHighlighter>
                                        </>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </SheetContent>
        </Sheet>
    );
} 
