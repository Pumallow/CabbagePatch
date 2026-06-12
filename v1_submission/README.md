This LLM derives its outputs off of the metadata, transcripts, tags, and other miscellaneous video data sourced from Youtube's API.
With the input in mind, the model compiles a list of 10 relevant videos then outputs the top 4-6 with reasonings as to why the user should watch this video and in which order to consume the material to reach said goal. 


src/Agent.py - 
- calls for the model generation
   - the model used: meta-llama/llama-3.1-8b-instruct
   - temperature = 0.3
- 2 wrappers for deepeval and llm for vectorizing
- truncation and sorting of youtube videos
- measuring computation costs

src/eval.py -
- Used a sklearn hashing vectorizer as opposed to hugging face due to context precision and recall no longer being free to calculate
- DeepEval used for faithfulness and answer relevancy

src/retrieval.py -
- Using youtube_transcript_api and googleapiclient.discovery to correctly extract data from Youtube opensource API.

test_set/test_inputs.json -
- offers 10 test cases to run the model. To expedite the learning process, the top 5 cases were used.


Here we compare 2 different prompts to see if the model will improve in performance:

Prompt #1:
"""
You are a YouTube study-planner agent.

You are given:
- A user goal and constraints.
- A list of retrieved YouTube videos in the `Facts` section, each with title, URL, duration, and transcript snippet.

Your job:

1. From ONLY the videos in `Facts`, select EXACTLY 4–6 videos.
   - Do NOT invent or hallucinate new videos.
   - Do NOT reference videos outside `Facts`.

2. For each selected video, provide:
   - Title
   - URL
   - Estimated role in the learning plan (e.g., "foundation", "project build", "deep dive").
   - Short reasoning (2–3 sentences) for why it was chosen.
   - A relevance score (0–10) and confidence score (0–10).

3. Rank the selected videos in order of contextual relevance to the goal.

4. Provide a textual walkthrough:
   - Describe how the user should use these videos over their time budget.
   - Tag videos inline (e.g., "[Video 2]") where they fit into the plan.

5. Briefly explain why the other videos in `Facts` were NOT selected:
   - Group them by reason (e.g., "too superficial", "off-topic", "shorts", "non-instructional").
Output format:

- First: a short summary of the plan (3–4 sentences).
- Then: a numbered list of the 4–6 selected videos with details.
- Then: a short section "Why other videos were not selected".
"""

Of 3 different prompts:
[{'faithfulness': 0.875, 'answer_relevancy': 0.6666666666666666}, {'faithfulness': 0.9375, 'answer_relevancy': 0.8421052631578947}, {'faithfulness': 1.0, 'answer_relevancy': 0.9375}]


Prompt #2:

"
You are a YouTube Study‑Planner Agent.

You receive:
- A user goal and constraints.
- A list of retrieved YouTube videos in the `Facts` section. Each video includes:
  - Title
  - URL
  - Duration (minutes)
  - Transcript snippet

Your task is to build a focused learning plan using ONLY the videos in `Facts`.

=====================
REQUIRED BEHAVIOR
=====================

1. Select EXACTLY 4–6 videos.
   - You may ONLY choose from the videos listed in `Facts`.
   - Do NOT invent videos, URLs, durations, or transcripts.
   - Do NOT reference videos outside `Facts`.

2. For each selected video, provide:
   - Title  
   - URL  
   - Estimated role (e.g., “foundation”, “deep dive”, “project build”)  
   - 2–3 sentence justification  
   - Relevance score (0–10)  
   - Confidence score (0–10)

3. Rank the selected videos from most to least relevant to the user goal.

4. Create a walkthrough plan describing how the user should study:
   - Reference videos inline using tags like `[Video 2]`.
   - Fit the plan within the user’s time budget.

5. Explain why the remaining videos were NOT selected.
   - Group them by reason (e.g., “off‑topic”, “too superficial”, “shorts”, “non‑instructional”).

=====================
OUTPUT FORMAT (STRICT)
=====================

1. Short Summary (3–4 sentences)

2. Selected Videos (Numbered List) 
   For each video:
   - Title  
   - URL  
   - Role  
   - Reasoning  
   - Relevance: X/10  
   - Confidence: X/10  

3. Walkthrough Plan

4. Why Other Videos Were Not Selected  
   - Category → list of video titles

Do not output anything outside this structure"

(FIRST 3 TESTS)
[{'faithfulness': 1.0, 'answer_relevancy': 0.6470588235294118}, {'faithfulness': 0.8333333333333334, 'answer_relevancy': 0.7368421052631579}, {'faithfulness': 0.8571428571428571, 'answer_relevancy': 0.6428571428571429}]


The second prompt for the LLM greatly improves the model in faithfulness due to its more established output framework. The prior prompt allowed for flexibility in how verbose the model could be but with the implementation of addons like the walkthrough plan and short summary, each containing detailed instructions, the model streamlined its outputs.
The answer relevancy remains quite steady despite the number of tests. Computational lag rests heavily on the evaluation methods however, pulling the data is quick.

Strengths:

The model does well combining the goals with the known inputs: 
----For test 3
Goal: Understand modern LLM architectures and inference optimizations
Unknown: Speculative decoding, KV caching, Quantization, LoRA adapters
1. **Video 1: LLM inference optimization: Architecture, KV cache and Flash attention**

The model also excels at takes heed to constraints at times:
--- With test 1
"Avoid pure-theory lectures and 'in 100 seconds' surface intros."
* **Shorts**: Video 6: When a Javascript developer discovers React for the first time...


Found Errors:
Halucinations -----
Input 2 requests a "slow-paced and beginner-friendly, avoiding advanced jargon"
Within the short summary its states: "The selected videos are designed to be slow-paced and beginner-friendly, avoiding advanced jargon."

The model then provides videos titled:
1. **Video 1: Do THIS instead of watching endless tutorials - how I’d learn Python FAST…**
2. **Video 6: How I Would Learn Python FAST (if I could start over)**


Final outputs also rendered the computational costs for each run (fractions of cents for the small scale given we are using Llama-3.1-8B):

[[{'faithfulness': 0.6153846153846154, 'answer_relevancy': 0.7083333333333334}, nan, nan, 0.0004946], [{'faithfulness': 0.7857142857142857, 'answer_relevancy': 0.8095238095238095}, nan, nan, 0.0005334000000000001], [{'faithfulness': 0.3333333333333333, 'answer_relevancy': 0.6666666666666666}, nan, nan, 0.0005244000000000001]]

Further Expansions:

I would resource other API taps like google search engines to find the most ideal websites with online educators and possibly quiz taking to ensure any material watched can then be tested on. Further testing can be done with more test sampling with stronger models to accommodate the larger work loads. A neutral juxtaposing LLM judge can be leveraged to unbiasely evaluate the outputs as well with deeper insight into the 4 metrics. Further analysis can also be done on latency and computational cost processes either through the evaluation on the OpenAI UI side or via the backend dictionary provided in outputs.

A deeper dive needs to be put on the constraints portions of the input. They should influence the model's approach to sorting the videos as opposed to key terminology or inquiring about the goal.

From a scalability approach, the use for each query will need to be optimized so chunking analysis should be implemented as well.
