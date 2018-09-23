# Lexis Nexis AntiTrust Analysis   

## Project studies the literature of Romance languages to establish and compare connotations of various forms of a single word or phrase in the Spanish language across regions of Latin America.
### The goal of the project is to see how various literary terms are used in modern contexts and if their spelling variations contribute any contextual differences. The hypothesis is that a word such as 'Don Quijote' has varying connotations in different regions of Latin America.
* data: Factiva Database

### Key-words
* tenorio, donjuan, burlador, don juan, quijotesco, quijotismo, quijotesca, quijote, quijotada, fuente ovejuna and fuenteovejuna
## Machine Learning Implementation

1. Vector Space Modelling (Word2Vec)
	1. A method used to create word accociations/embeddings based on word usage across text documents.
	2. Word2Vec is a two-layer neural network python library that is trained to recustruct context of words  
2. Tensorboard
	1. Tensorboard, a tensorflow library, is used to visualze the multi-dimensional word embedding corpus
3. Topic Modelling (Latent Dirichlet Allocation)
	1. A statistical model for discovering abstract topics that occur in a collection of documents
	2. Create topic models for the key-word variations