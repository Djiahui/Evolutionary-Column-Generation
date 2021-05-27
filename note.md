# Decoder for SPPTW
Motivated by labeling algorithm, for each customer in pre-constructed unreachable
resource, thus, for each node a set which contains unreachable customers should be maintained.

## relationship matrix
when the constructed route need to decide the next customer,
we can design a function to calculate the possibility of customers to be selected.

During the decoding, we sample a customer from the reachable customers of current node
,if the selected target is infeasible, add it into unreachable customers set and sample another one from candidates. If there exists no feasible customer
or the depot is selected, then the route is constructed.
## tree search
Obviously, simple tree search is not enough...