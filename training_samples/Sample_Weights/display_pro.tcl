mol delrep 0 top

mol representation VDW 1.7 20
mol color ColorID 22
mol selection {resname PAM}
mol addrep top

mol representation VDW 1.7 20
mol color ColorID 4
mol selection {resname PAA}
mol addrep top

mol representation VDW 1.7 20
mol color ColorID 30
mol selection {resname TBA}
mol addrep top

mol representation VDW 1.7 20
mol color ColorID 3
mol selection {index 123 to 656}
mol addrep top

pbc box 
#pbc box -color blue
#pbc box -style tubes
#pbc box -width 3
#pbc box -resolution 12

color Display Background white
display depthcue off
axes location off

rotate y by 84
rotate x by 14
rotate z by 1.2
clear
