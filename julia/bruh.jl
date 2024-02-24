using Flux, ProgressMeter

struct Bruh
  weights
  biasies
  Bruh() = new(rand(3,3), rand(1,3))
end

function (b::Bruh)(x)
  return  x * b.weights + b.biasies
end

Flux.@functor Bruh

model = Bruh()
optim = Flux.setup(Flux.Adam(0.01), model)
losses = []

@showprogress for epoch in 1:1_000
  x = rand(1,3)
  y = rand(1,3)
  loss, grads = Flux.withgradient(model) do m
    y_hat = m(x)
    Flux.crossentropy(y_hat, y)
  end
  Flux.update!(optim, model, grads[1])
  push!(losses, loss)  # logging, outside gradient context

  # sleep for 200 milliseconds
end

println(losses[1])
println(losses[end])
