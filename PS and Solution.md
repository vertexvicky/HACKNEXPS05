this is our problem statement:
HNX26EPS05: Multi-Stream Video Intelligence with Conversational Query
Pitch: Plug in multiple CCTV streams; the system continuously detects and indexes events across all
cameras, and exposes a chat interface where a user describes what they want ("did a red car pass
through the main gate in the last hour?") and gets back which camera, when, and the visual
evidence.
Why it's hard: Four separable challenges:
• Open-vocabulary search. Queries like "red car" or "person carrying a large bag" won't match a
fixed label set. This needs open-vocab detection or vision-language embeddings, not a closed
class list.
• Grounded, localized answers. Every answer resolves to a specific camera + timestamp + visual
evidence. No traceable source = scored failure.
• Clarify-once, then remember. When the system doesn't know a referent ("which one is the
main gate?"), it asks, lets the user point to the camera/region, and then persists that fact
permanently. It must never ask again, even after a restart. This is a small learned knowledge
base, not just chat history.
• Cross-camera continuity. Re-identify the same car/person across non-overlapping cameras
and reconstruct its path ("Gate cam 09:14 → lobby cam 09:16 → rear cam 09:22").
Baseline to beat (chosen by the team): The strongest option of this kind available today, chosen by
the team (for example a standard open-vocabulary detection plus embedding-retrieval pipeline), run
on the same footage.
Minimum bar to qualify (24h): Use recorded multi-camera footage (not live). Support
natural-language queries that return the correct camera + timestamp + clip, beating the baseline on
retrieval accuracy for the held-out queries, with the clarify-once memory behavior, plus a short
write-up.
Stretch goals:
• True live multi-stream ingestion.
• Cross-camera re-ID and timeline reconstruction.
• Standing queries / real-time alerts ("notify me if anyone enters this zone after 8pm").
• Privacy handling (face blurring, on-prem-only inference).
Data and judging: Teams use their own multi-camera footage, recorded or taken from public
datasets. At judging, judges bring unseen recorded multi-camera footage and natural-language
queries with known answers.
Judging rubric:
• Retrieval accuracy on held-out NL queries vs. baseline (30%)
• Correctness of camera + timestamp localization (20%)
• Clarify-once memory: asks appropriately, never re-asks, persists across restart (20%) • Query latency vs. baseline (10%)
• Research contribution: what is new compared with the baseline, shown by an ablation or
comparison (20%)
• Bonus: live ingestion, cross-camera tracking, alerts, privacy
Skills: video decoding pipelines, open-vocab / VLM detection, embedding retrieval + temporal
indexing, multi-object tracking + re-ID, persistent memory design.


this our current proposed solution:
We will input two videos 
the model contains a extreme lightweight vehicle dection ( tracking ) + number plate detection then OCR 

Thej it has face detection ( even smaller pixel ) then embbdedd and recognition ( but the face recognition with its existing DB . then store the embedded and group the embedded under unknown___ 

theh if in a continues frame face it is not recognised then we will body and outfit embeddings to track the person 

also the main thing is I wana track the object twoo its surrounding . 

example track the bag that was carried by { name } 


theh we need to make more inteleccutal because name santhya already have a face but what we need is we have track all the relation 


basically we have to track all the relation for everything . 

it not should be a model or a product that will directly retive by keyword it should but by action .

also it should be lightweight . 

I should be a reasoning and understanding model not a keyword matching platform