> **See also: [[INDEX]]** — the full depth map over this vault (how far each file goes, what it settles, what it
> kills, where the knowledge stops), plus the merged Closed / Pending / Open status of the field. This file below
> is my own one-line reminder per doc; the index is the detailed version.

## Notes

> The reminder for each docs

1. [[1-transformer]] : I already understand all part deeply. Now lacking only deep insight how weight activation inside work. In the future need more to go deep in each transformer part such as multi-head theory, multi block theory, optimizer, convergence prove, etc.
2. [[2-hopfield]] : I know about its concept now, didn't deep drive into each formular yet. Require more to understand how hopfield really is (Now I just know what it look like, but not have deep understand how it really work in every part). And in far future, require to prove this concept formular myself.
3. [[3-transformer-internal]] : Tell more insight about how transformer really is. Need more time in the future to build my final mental intuition about it. I just feel I'm nearing understand it all.
4. [[4-hopfield-internal]] : Tell more insight about how hopfield2020, hopfield1982, hopfieldPooling, and hopfieldLayer really is. Still need more time to understand energy or jacobain formular. 
5. [[5-interp-hull-and-superposition]] : Tell about convex hull (where the model
   can go) and superposition (how much it can hold) to prove attention and FFN. Still need more time to deep drive into Gram matrix and the interp concept.
6. [[6-fixed-point-and-what-transformer-chasing]] : Tell more about hopfield landscape, what it really is, till the last piece to connect hopfield to transformer and what attention chase. Need more careful read in the future to make final complete intuition.
7. [[7-vit-foundation]] : Introduce about how we apply transformer into image. (Main direction is use all transformer to see it limit). Need to understand more how transformer really do on image, how it understand spartial, how we embedded position, etc. (Especially how attention really do on this datas)
8. [[8-position-transformer]] : Deep drive into how position really is, how it really work, etc. Need more time to understand each position embedded formula
9. [[9-vision-tranformer-landcape]] : Tell more how other paper improve vit. Separate image model to 3 part and adjust it. Need more time to read each part carefully how current paper reach.
10. [[10-attention-collapse-and-field-equilibrium]] : Tell more about what is really tokens mean and be in attention. How it collapse into N\*d size different from hopfieldPooling that collapse to 1\*d. And tell more about the dynamic latent space in stop when equilibrium concept. In the future need more time to deep drive into real dynamic latent space paper. 
11. [[11-perciever-and-more]] : Tell about how we make q size to "more" or "less" than kv. How encode, and decode really work. And how can we reduce attention size while keep accuracy. Need more time to deep drive into dynamic latent space and movable attention concept.
12. [[12-recurrent-visual-attention]] : Deep drive into full mechanism of glimpse-based attention. Start from 2012 that use reinforce derivation until now that use STN, DRAW, Gumbel, etc. about why it lost, and what survived as pointer-based reads. Need more time to understand full attention image how it really be and how to implement of this complex system.
13. [[13-latent-space-and-shortcut]] : Show more insight about how latent space really is. Especially to connect it to image type. And tell more about the shortcut that destroy latent space. Need more time to carefully understand latent space, how it really be, how to design it, and is it really important?

## Ideas
1. [[s1-opened-topic-ideas]] : The ideas after understand 11 notes. 
	1. Object-bound dynamic latent
	2. Query-conditioned compaction
2. [[s2-opened-topic-ideas]] : Bounded-bandwidth visual reasoning, The ideas after understand 13 notes. 
	1. Latent predicts position AND scale (l, s) 
	2. Hopfield-style address retrieval for gaze 
	3. Pointer-chase deep-hard benchmark

