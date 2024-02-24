using Flux, ProgressMeter

# DAG where nodes dont have any attributes
struct Node
  edges::Vector{Int}
end

struct BiNode
  out::Node
  in::Node
end

struct Graph
  nodes::Vector{BiNode}
end

struct Encoder
  Embeddings::Matrix{Real}
  Wq::Matrix{Real}
  Wk::Matrix{Real}
  Wv::Matrix{Real}
  P::Matrix{Real}
  function Encoder(MaxNodes, EmbeddingSize, AttentionSize)
    Embeddings = rand(MaxNodes, EmbeddingSize)
    Wq = rand(EmbeddingSize, AttentionSize)
    Wk = rand(EmbeddingSize, AttentionSize)
    Wv = rand(EmbeddingSize, AttentionSize)
    P = rand(AttentionSize, EmbeddingSize)
    new(Embeddings, Wq, Wk, Wv, P)
  end
end

function (encoder::Encoder)(node::Node)::Matrix{Float64}
  emb = [encoder.Embeddings[i, j] for i in node.edges, j in 1:size(encoder.Embeddings, 2)]
  q = emb * encoder.Wq
  k = emb * encoder.Wk
  v = emb * encoder.Wv
  scores::Matrix{Float64} = q * k'
  attention = softmax(scores)
  attention * v * encoder.P
end

Flux.@functor Encoder

model = Encoder(10, 5, 5)
optim = Flux.setup(Flux.Adam(0.01), model)
losses = []

@showprogress for epoch in 1:1_000
  x = max.(rand(Int, 10) .% 10, 1)
  y = rand(10, 5)
  #println(size(y_hat))
  loss, grads = Flux.withgradient(model) do m
    y_hat = m(Node(x))
    Flux.crossentropy(y_hat, y)
  end
  #Flux.update!(optim, model, grads[1])
  push!(losses, loss)  # logging, outside gradient context
end

println(losses[1])
println(losses[end])
