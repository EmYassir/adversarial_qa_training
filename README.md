## Monthly Progress

| Assignee                      | Deadline   | Progress                                                                 | Description                                |
|-------------------------------|------------|--------------------------------------------------------------------------|--------------------------------------------|
| Oussama             | 2022-05-20 | ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/100) | Litrature review  and brainstorming                      |
| Yassir             | 2022-05-20 | ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/100) | Litrature review  and brainstorming                      |

## Research Proposal: Adversarial Answer Oriented Passage Re-Ranking

* **Task** Question Answering

* **Problem statement** QA models can accurately extract the correct answer span but most of the time from an irrelevant Paragraph.  

* **Project Objective**
    * [ ] Run the current baseline on large datasets and improve it with BERT.
    * [ ] Upgrade the GAN framework to mask-edit generative GAN.
    * [ ] Annotate a subset of challenging <contain answer, irrelevant> passages and advocate the evaluation on it.  
     
* **A top-tier conference paper is publishable if**: 
    * [ ] BERT improves the baseline compared to RNN encoder.
    * [ ] Our new GAN approach significantly outperform the improved baseline.
    * [ ] We distribute a dataset with challenging <contain answer, irrelevant> passages, where we show that: current systems perform poorly, our approach works much better.  

### Here is the roadmap of the project [May-September]

* [ ] Implement the baseline [2-3 week]
    * [ ] Run the base code and reproduce the paper results.
    * [ ] Run the code on large scale datasets.
    * [ ] Refactor the code: 
        * [ ] New dataloader.
        * [ ] Dense Retrieval tools (both BM25, DPR\FAISS)
        * [ ] Automatic Fine-tune and evaluate pipeline. 

* [ ] Improve the current baseline [1-2.5 week]
    * [ ] Replace RNN by BERT encoder.
    * [ ] Test different Models type (Splinter, Roberta, SpanBERT) and sizes (small, base, large).
    * [ ] Generate benchmarking results model size vs. latency.

* [ ] Explore 1: Post-Reranking with GAN [2-3 week]
    * [ ] Replace the MRC generator by a re-ranker classifier.
    * [ ] Place the GAN framework after the reader and compare with previous approach.
    * [ ] Goal: significant improvement in term of latency and performance compared to the baseline.

* [ ] Explore 2: Adversarial DA with GAN [3-5 weeks]
    * [ ] Develop methods to select Question-Passage tokens that mostly influence model (heuristic, random, attention).  
    * [ ] Explore the mask-and-edit adversarial  approach (MATE-KD) for getting hard <contain answer-irrelevant> examples.
    * [ ] Explore the mask-and-edit adversarial  approach (MATE-KD) for getting hard <not contain answer-relevant> examples.
    * [ ] (optional) DA methods for hard examples generation (QG).

* [ ] Explore 3: Dataset Building [2.5-3.5 weeks]
    * [ ] Get an off-the-shelf-retrieval and Ranker model.
    * [ ] Predict on dev set of multiple datasets and select the one(s) that contains hard  <contain answer-irrelevant> examples.
    * [ ] Develop a semi-automatic method to filter down the hard <contain answer-irrelevant> examples.
    * [ ] Design an annotation task and assign it to annotator.
    * [ ] Annotate (done by external annotator).
    * [ ] Ensure that the sub select set is challenging for systems.

* **Reference List**
    * PReGAN: Answer Oriented Passage Ranking with Weakly
Supervised GAN
    * [MATE-KD: Masked Adversarial TExt, a Companion to Knowledge
Distillation](https://arxiv.org/pdf/2105.05912.pdf)


## How to run the code (RANKER):
Note that in steps 1-4, you will have to change the hard coded paths to the desired dataset, embedding, and output directory.
0) Download Stanford CoreNLP from https://stanfordnlp.github.io/CoreNLP/
1) Run normdata.py 
2) Run pre_docrank.py
3) Run pre_txt_token.py
4) Run trainranker.py