% already a link
link(node(bfactor(1)), node(afactor(1))), link(node(bfactor(7)), node(afactor(7))), link(node(bfactor(4)), node(afactor(2))), link(node(bfactor(5)), node(afactor(2))), link(node(bfactor(6)), node(afactor(2))), link(node(bfactor(2)), node(afactor(3))), link(node(bfactor(3)), node(afactor(3))).

% example 1
entail(query, node(bfactor(1))), entail(query, node(bfactor(7))), false.

% example 2
entail(query, node(bfactor(3))), entail(query, node(bfactor(7))), false.

% example 3
entail(query, node(bfactor(1))), true.

% example 4
entail(query, node(bfactor(1))), entail(query, node(bfactor(2))), true.

% example 5
entail(query, node(bfactor(1))), entail(query, node(bfactor(2))), entail(query, node(bfactor(7))), true.

% example 6
entail(query, node(bfactor(1))), entail(query, node(bfactor(2))), entail(query, node(bfactor(7))), entail(query, node(bfactor(6))), true.