tree-kernel
=========

 [**KeLP**][kelp-site] is the Kernel-based Learning Platform developed in the [Semantic Analytics Group][sag-site] of
the [University of Roma Tor Vergata][uniroma2-site].

This is the tree-kernel module of KeLP. It includes various some convolution kernels:

**DIRECT KERNELS ON TREES:**

* _SubTreeKernel_: it is the tree kernel described in (Vishwanathan '03) and optimized in (Moschitti '06a). It operates on _TreeRepresentations_ evaluating the number of common fragments shared by two trees. 
The considered fragments are complete subtrees, i.e. a node and its entire descendancy. 

* _SubSetTreeKernel_: it is the tree kernel described in (Vishwanathan '03) and optimized in (Moschitti '06a). It operates on _TreeRepresentations_ evaluating the number of common fragments shared by two trees. The considered fragments are are subset-trees, i.e. a node and its partial descendancy (the descendancy can be incomplete in depth, but no partial productions are allowed; in other words given a node either all its children or none of them must be considered).

* _PartialTreeKernel_: it is the tree kernel described in (Moschitti '06b). It operates on _TreeRepresentations_ evaluating the number of common fragments shared by two trees. 
The considered fragments are partial subtrees, i.e. a node and its partial descendancy (i.e. partial productions are allowed).

**DIRECT KERNELS ON SEQUENCES:**

* _SequenceKernel_: it is the sequence kernel described in (Bunescu '06). It operates on _SequenceRepresentations_ evaluating the common subsequences between two sequences.


============
REFERENCES:

(Bunescu '06) Razvan Bunescu and Raymond Mooney. _Subsequence kernels for relation extraction_. In Y. Weiss, B. Scholkopf, and J. Platt, editors, Advances in Neural Information Processing Systems 18, pages 171-178. MIT Press, Cambridge, MA, 2006.

(Moschitti '06a) Alessandro Moschitti. _Making Tree Kernels Practical for Natural Language Learning_. In Proceedings of EACL, 2006

(Moschitti '06b) Alessandro Moschitti. _Efficient convolution kernels for dependency and constituent syntactic trees_. In proceeding of European Conference on Machine Learning (ECML) (2006)

(Vishwanathan '03) S.V.N. Vishwanathan and A.J. Smola. _Fast kernels on strings and trees_. In Proceedings of Neural Information Processing Systems, 2003.

[sag-site]: http://sag.art.uniroma2.it "SAG site"
[uniroma2-site]: http://www.uniroma2.it "University of Roma Tor Vergata"
[kelp-site]: http://sag.art.uniroma2.it/demo-software/kelp/

