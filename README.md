# Private knowledge matching: finding answers to questions using private set intersections and LLMs
This is going to be the core repo for this project which demonstrates the capabilities. 

## Abstract
There are many instances where two parties share common knowledge and want to know what they share, but are unwilling to share all the information upfront over concerns of privacy or information advantage. This common knowledge can be answers to key questions, shared preferences, or unexpressed challenges. These pieces of information can be expressed in natural language and encapsulated using language model text embeddings. This work explores how these two parties can identify this shared knowledge, without needing to share upfront text or embedding data, by using private set intersections over text embeddings that are binned into hypercubes that decrease in size at better matches are found. This approach can match the accuracy of non-privatized embedding matching within 90%. Further, we show how this approach can be extended to two LLMs matching on shared knowledge, through repeated generation, embedding, and reverse embedding of textual information from the LLMs. This presents a new paradigm through which two agents can determine shared information without needing to reveal any of their knowledge upfront.

## The Plan for The Project:
We show that PSI can effectively match questions with answers using iteratively reduced spatial hashes without leaking information on questions or answers that are not shared.
We extend this idea to two LLMs trying to match on answers to questions, 

### How, part 1:
- Find a question-answering dataset / benchmarking. Run sbert / ada embeddings on this benchmark.
- Embed both the queries and the documents using the same embedding.
- Bin the embeddings in chunks (essentially round them to a decimal point).
- Match on the highest level of the embedding; those the match will be subdivided and the process is repeated.

### How, part 2:
- We ask Claude or OpenAI to pretend to be an expert in an area. We generate a bunch of questions. 
- We sbert / ada embed and bin those questions. 
- We get the other AI to generate some possible topics it’s willing to talk about at a high level. We sbert embed and bin those questions. 
- We match on these topics and reveal the bin (no detailed answers are released). 
- We sample from across the bin to get vectors and create text using vec2text
- Each LLM is asked to generate questions related to this topic. 
- We repeat using smaller bins until we have the answers we need. 
- The LLM can then send the raw text answers to the asker knowing that they have utility for one another. 

## Repo Explainer
Part 1 of the project will be contained in `knowledge_matching` and part 2 will be contained in `agents_talking`. Read their sub-READMEs for more information.


### Requirements
We're going to need:
* [OpenMined's PSI](https://github.com/OpenMined/PSI) (install via [PyPi](https://pypi.org/project/openmined.psi/)) for PSI.
* [SBERT](https://www.sbert.net/) for embeddings.
* [Vec2Text](https://github.com/jxmorris12/vec2text).
* [OpenAI's Ada Embeddings](https://platform.openai.com/docs/guides/embeddings) for easy embeddings.
* [OpenAI's GPT-3](https://platform.openai.com/docs/api-reference) for LLMs talking.

