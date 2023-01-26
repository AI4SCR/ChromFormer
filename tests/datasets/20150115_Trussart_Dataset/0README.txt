============= 2015-01-15

This dataset corresponds to the data used in the paper:

Assessing the limits of restraint-based 3D modeling of genomes and genomic domains.
Marie Trussart, Francois Serra, Davide Bau, Ivan Junier, Luis Serrano and Marc A. Marti-Renom
2015

Directories:
	scripts 		--> contains the script to calculate the MMP score.
		    	    		usage: mmp_score.py -i "inputmatrix" -o "outputdirectory"
	Real_HiC 		--> contains the three input real Hi-C matrices described in Figure 5 panel C.
	Toy_Models 		--> contains all the toy models used to generate the Simulated_HiC matrices organized in six folders (one per genomic architecture)
	Simulated_HiC 		--> contains the 168 simulated Hi-C matrices organized in six folders (one per genomic architecture).
	Reconstructed_Models 	—-> contains the 168 best reconstructed models from IMP organized in six folders (one per genomic architecture).
				    models are stored in two different formats, XYZ (text based) and CMM (Chimera format, http://www.cgl.ucsf.edu/chimera/)
