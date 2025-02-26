link(support(bfactor(1), root(1))).
link(attack(group(bfactor(7), afactor(2)),support(bfactor(1), root(1)))).
link(attack(afactor(3), attack(group(bfactor(7), afactor(2)),support(bfactor(1), root(1))))).
link(support(bfactor(2), afactor(3))).
link(support(bfactor(3), afactor(3))).
link(support(bfactor(4)), afactor(2)).
link(support(bfactor(5)), afactor(2)).
link(support(bfactor(6)), afactor(2)).

entail(query, bfactor(1)).
entail(query, bfactor(6)).
entail(query, bfactor(7)).