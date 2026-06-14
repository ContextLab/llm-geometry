# Overview

This project is about understanding LLMs using geometric approaches and visualizations.  Everything should be implemented as interactive webpages with beautiful smooth animations and responsive interactions.  If anything needs to be precomputed, it should be done once and cached, with a smooth animation showing progress while that happens. The entire implementation should look modern, striking, beautiful, clean, and intuitive.  Everything should be clearly explained and documented.

# Approach

There are three core approaches: vector fields, sankey diagrams, and manifold visualizations.

## 1. Transformer layers as vector fields

Each token's embedding provides a location in the model's embedding space. Given all tokens observed thus far *prior* to token $t$, if token $t$ is set (in turn) to each *possible* token, it will result in some *next* token, $t + 1$. We can draw a vector starting at token $t$'s embedding coordinate, and ending at token $t + 1$'s embedding coordinate.

We can also visualize this:
 - Map (using UMAP or PCA) embedding space into 2D
 - Construct an $n$ by $n$ rectangular grid of embedding coordinates
 - For each grid vertex, find the closest token to that coordinate. These are the *reference points*.
 - Now we can construct a quiver plot showing where each token leads next

Interesting variants or approaches:
 - Interactive plotting: hovering over any reference point or vector (arrow) reveals the corresponding token(s)
 - Animations/sliders: select which layer(s) is/are shown (default: input at the first layer, "response" is from last layer)
 - Interactive "history" or "context": optionally, add in a prompt to prepend to the input before the "first" (originating) token. (User can type in anything they wish, or select from a dropdown menu.)
 - Interactive "response" as a trajectory: given some text, show the "path" through embedding space (color denotes probability of each token), and/or create an animation showing how the vector fields change with each subsequent token in the sequence
 - Ability to adjust temperature with a slider or text box; temperature > 0 results in *multiple* vectors (more transparent = lower probability; distribution estimated with 100 or so repetitions) originating from each reference point
 - Dropdown menu for selecting different models, and/or adding arbitrary open weights models from HuggingFace

## 2. Token sequences as sankey diagrams

For a given (arbitrary) sequence, use a particle swarm to estimate the probability of emitting each possible token, at each subsequent position in the sequence. X-axis: time (position in the sequence); Y-axis: token ID

In other words:
- Start with a "prompt" (user selects from a dropdown or types in something arbitrary)
- The first "response" token is at time = 0. Compute probability of emitting each possible token first (note: this needs to be done with an open weights model where we have access to this information). Display the distribution in some sort of visually appealing way (violin plot?) at that position on the x-axis. Now take $n$ samples from that distribution. These are the particles.
- At time = 1, each *particle* will now treat its previous draw as observed, and compute the probabilities of emotting each possible *next* token (at time = 2). Combine across particles (average? multiply and re-normalize?) to compute the full distribution and visualize in an appealing way. Now take actual draws from each particle's next distribution to get observed $t = 2$ tokens.
- Repeat until target sequence length is reached. If any particle emits an end of stream token, no further samples are taken from that particle after that timepoint.

Interesting variants or approaches:
  - Interactive plotting: hover over any stream or particle to see its tokens
  - Interactive "history" or "context": optionally, add in a prompt to prepend to the input before the "first" (originating) token. (User can type in anything they wish, or select from a dropdown menu.)
  - Interactive "response" as a trajectory: given some text, show the "path" through embedding space (color denotes probability of each token), and/or create an animation showing how the vector fields change with each subsequent token in the sequence
  - Ability to adjust temperature with a slider or text box; temperature > 0 results in *multiple* vectors (more transparent = lower probability; distribution estimated with 100 or so repetitions) originating from each reference point
  - Dropdown menu for selecting different models, and/or adding arbitrary open weights models from HuggingFace

## 3. Reachable "thoughts" as a manifold

For this approach we need to reduce the embeddings to 3D spherical coordinates (using spherical MDS or similar).

We start with a unit sphere in 3D. Now plot the embeddings of all possible tokens on a sphere of radius 2.

Given a distribution of probabilties of emitting each possible token, we can morph the sphere as follows:
- Find the closest point on the surface of the sphere to each possible target token. These are just the embedding coordinates on the unit sphere.
- Move that closest point towards the target token's embedding coordinate on the radius 2 sphere. Distance is proportional to probability of emitting that target token next.
- We also warp *other* coordinates on the unit sphere towards the target coordinate, using RBF interpolation. Something like:
```python
def normed_rbf(x, center, width, exponent=2):
    vals = np.exp(-np.divide(np.power(cdist(x, np.atleast_2d(np.array(center))), exponent), width))
    vals -= np.min(vals)
    vals /= np.max(vals)
    return vals

def warp_mesh(mesh, target, p, width, exponent=2):
    #mesh: the surface
    #target: where to warp towards
    #p: proportion of the distance from closest point to target that the closest point should move
    #width: RBF width parameter
    
    #get closest point to target
    dists = cdist(np.asarray(mesh.vertices), np.atleast_2d(np.array(target)))
    closest = int(np.where(dists == np.min(dists))[0][0])
    
    #compute RBF weights usng closest vertex
    weights = np.multiply(p, normed_rbf(np.asarray(mesh.vertices), mesh.vertices[closest], width, exponent=exponent))
    
    #warp each point towards the target according to the RBF weights
    x = np.asarray(mesh.vertices)
    target = np.tile(target, [x.shape[0], 1])
    weights = np.tile(weights, [1, x.shape[1]])
    
    warped = np.multiply(target, weights) + np.multiply(x, 1 - weights)
    
    ids = o3d.utility.IntVector(list(range(x.shape[0])))
    positions = o3d.utility.Vector3dVector([c.flatten() for c in np.split(warped, warped.shape[0], axis=0)])
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info) as cm:
        deformed = mesh.deform_as_rigid_as_possible(
            ids, positions, max_iter=5)
    
    deformed.compute_vertex_normals()
    
    return deformed
```
- Loop over all possible tokens to get the final distortion. Note: I'm not sure how to combine so that we can get the same answer regardless of looping order over tokens. Need to research this.
- Now we can use this deformed surface to visualize probability of emitting each possible next token.
- We can pursue analogous ideas to with the vector field (quiver plot) way of visualizing probabilities.
- We'll need a way to smoothly interact (rotate, pan, zoom, etc.) with the manifold, or visualize how it changes over time (in an animation), etc.
