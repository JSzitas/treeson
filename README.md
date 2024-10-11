# treeson

Simple, single header decision tree/random forest library. Intended to enable embedding ad-hoc data in terminal nodes of trees, creating custom splitters, and offering all this at very little performance loss. 

Supports stateless prediction (predict from a random forest without ever fully storing it, in memory or otherwise), larger than memory models, serialising and deserialising to disk as needed.
Also supports generating predictions from serialised models.

Your RAM does not really matter, you will never run out of memory using this library, unless you really want to. 

Also, obviously, multithreaded.

See examples for example usage. 
