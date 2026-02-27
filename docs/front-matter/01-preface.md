# Preface

## Why This Book Exists

The field of Artificial Intelligence and Machine Learning has exploded in scope and complexity. When I began my journey as an AI/ML engineer, there was no single resource that covered the full breadth of what a modern practitioner needs to know — from the mathematical foundations that underpin every algorithm to the distributed systems that train models with billions of parameters, from classical machine learning to the frontier of agentic AI systems.

Textbooks covered theory but not practice. Blog posts covered practice but not depth. Online courses covered breadth but not the connective tissue that binds everything together. And none of them told you *what order to learn things in*, which is arguably the most important decision a learner makes.

This book is my attempt to fix that.

## Who This Book Is For

This book is written for three audiences:

1. **Students** who are studying computer science, data science, or a related field and want a structured path into AI/ML engineering. You may have taken a machine learning course, but you're not sure how to go from homework assignments to production systems.

2. **Career transitioners** who come from software engineering, data engineering, analytics, or adjacent fields. You can code. You understand systems. But the ML landscape feels vast and disorienting.

3. **Practicing engineers** who want to level up from mid-level to senior. You've trained models and deployed them, but you know there are gaps — maybe in distributed training, maybe in the mathematics, maybe in LLM internals. This book maps those gaps and fills them.

This is *not* a book for complete beginners in programming. I assume you can write Python, use the command line, and have encountered basic programming concepts. If you're truly starting from zero, I recommend completing a Python fundamentals course first and then returning here.

## How This Book Is Organized

The book follows a carefully sequenced **seven-phase roadmap** that I developed through years of practice, mentoring, and hiring in the AI/ML space. The phases build on each other:

- **Part I: Foundations** (Chapters 1–4) covers Python mastery and the mathematics that every ML algorithm is built upon — linear algebra, calculus, probability, and statistics.

- **Part II: Classical Machine Learning** (Chapters 5–7) covers the algorithms and frameworks that remain the backbone of industry ML — from logistic regression to gradient boosting to PyTorch fundamentals.

- **Part III: Large Language Models** (Chapters 8–11) takes you from fine-tuning pretrained models to understanding how they are built from scratch, including alignment techniques like RLHF and DPO.

- **Part IV: Distributed Training & Performance** (Chapters 12–14) covers how models are trained across multiple GPUs and machines, and how to optimize every stage of the pipeline.

- **Part V: Generative AI & Advanced Applications** (Chapters 15–18) covers RAG systems, agentic AI, multimodal models, and production LLM serving.

- **Part VI: MLOps & Infrastructure** (Chapters 19–21) covers experiment tracking, data engineering, cloud platforms, and everything needed to take ML from notebooks to production.

- **Part VII: Frontier Topics** (Chapters 22–26) covers specialized and emerging areas — advanced computer vision, reinforcement learning, graph neural networks, time series, and responsible AI.

Each chapter includes:
- **Clear explanations** of concepts, building from intuition to formal definitions
- **Mathematical foundations** where appropriate, with step-by-step derivations
- **Code examples** in Python using industry-standard libraries
- **Practical exercises** to reinforce understanding
- **References and citations** to original papers and authoritative sources

## How to Read This Book

**If you are a student or beginner:** Read the book in order, from Part I through Part VII. Do not skip the foundations even if you think you know them. Depth in fundamentals is what separates senior engineers from mid-level ones.

**If you are an experienced practitioner:** Use the table of contents to identify your gaps. Each chapter is designed to be relatively self-contained, with clear prerequisites listed at the beginning. You can jump to the topics you need.

**If you are preparing for interviews:** Focus on Parts I–III and Chapter 15 (RAG systems). These cover the vast majority of what is asked in AI/ML engineering interviews at top companies.

**For everyone:** Do the exercises. Read the cited papers. Run the code. The single fastest way to learn is to find the GitHub repository for a technique, run the example, and then modify it for your own use case. Reading alone does not build working knowledge.

## A Note on Citations and References

This book draws on hundreds of research papers, textbooks, blog posts, official documentation pages, and open-source projects. I have made every effort to cite original sources accurately. Citations follow a modified APA format and are collected at the end of each chapter as well as in a comprehensive bibliography in Appendix D.

If I have inadvertently failed to credit a source or misrepresented any work, please contact me so I can correct future editions.

## Acknowledgments

This book would not exist without the extraordinary open-source and open-research communities that have made AI/ML knowledge freely available to the world. I am particularly indebted to:

- The authors of the foundational papers cited throughout this text
- The teams behind PyTorch, Hugging Face, scikit-learn, and the countless open-source tools that make this field accessible
- The educators who create free content — Andrej Karpathy, Jeremy Howard, Grant Sanderson (3Blue1Brown), Josh Starmer (StatQuest), and many others
- The AI/ML communities on Hugging Face Discord, EleutherAI, Reddit r/MachineLearning, and Twitter/X who share knowledge daily
- Every student and mentee whose questions shaped the structure of this roadmap

And most importantly, to every reader who picks up this book with the intention of learning. Your curiosity is what drives this field forward.

---

*February 2026*

