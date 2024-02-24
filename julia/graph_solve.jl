f(x) = 2x+3
sample = [3, f(3)]

@show sample

struct Node
  op
end

struct Constraint
  node
  value
end

struct Given
  node
  value
end

struct Graph
  nodes::Dict{Symbol, Node}
end



constraint(Eq, 2x + 3, 3)

Sum(a*const(x), b) == 9
a = (3+b)/x

assume(x == 3)
assume(b == 3)

solve(a)
  0: 3
  +3: 9-3 = 6
  *3: 6/3 = 2

# Bounds - const dep
0 => a*x # We call this a sub-expression
1 => :0 + b

## Aditional arbitraty constraint a & b > 0

### Training
a*3+b == 9
  b <= 9 && 9-:0
  a <= 3 && 9-b

a*4+b == 10
  b <= 10  && 10-:0
  a <= 2.5 && 10-b

### Result
b <= 9
a <= 2.5

b >= 1.5 # (9 - 2.5 * 3)
a >= 0

# Scales - lin dep
0 => a*x + b*y + c*z

## Aditional arbitraty constraint a&b>0 && x=z

### Training
a*3 + b*3 + c*3 == 9
  a <= 3 && 9-(b*y + c*z)
  b <= 3 && 9-(a*x + c*z)
  c <= 3 && 9-(b*y + a*x)

a*3 + b*4 + c*3 == 10
  a <= 3   && 10-(b*y + c*z)
  b <= 2.5 && 10-(a*x + c*z)
  c <= 3   && 10-(b*y + a*x)

### Result
# from b, we can calculate lower bound for a, which also implies an upper bound for c, because their are linearly dependent.

a <= 3
b <= 2.5
c <= 3

# 9-(2.5*3 + 3*3) = -7.5
# This becomes further complicated, when sub-expressions are non-linear. As the bounds to test against are non-trivial.
a >= 
