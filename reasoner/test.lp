link(support(node(bfactor(1)), node(root(1)))).
link(attack(node(bfactor(7)),node(bfactor(1)))).
link(attack(node(afactor(2)),node(bfactor(1)))).
link(attack(node(afactor(3)),node(bfactor(7)))).
link(attack(node(afactor(3)),node(bfactor(2)))).
link(support(node(bfactor(2)), node(afactor(3)))).
link(support(node(bfactor(3)), node(afactor(3)))).
link(support(node(bfactor(4)), node(afactor(2)))).
link(support(node(bfactor(5)), node(afactor(2)))).
link(support(node(bfactor(6)), node(afactor(2)))).

entail(query, node(bfactor(1))).
entail(query, node(bfactor(6))).
entail(query, node(bfactor(7))).