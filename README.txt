Since the presentation on March 1st, we have made some changes:
•	We added two levels, 9 and 10, with grids of 100x100 and 150x150, respectively.
•	We added some features to the generate_positions function with the aim of creating a list of obstacles for levels 9 and 10:

     o	We inserted the is_adjacent function to prevent each obstacle from having more than one adjacent to it.
     o	We added the generate_obstacles function that generates the obstacles.
     o	When levels 9 or 10 are selected, we replaced 3 and 1 obstacles, respectively, to improve the flow of the game.

•	We added the visualization of level 9 (100x100) to the demo.
•	The delays were commented for faster visualization of the results, that is why the easier level are much faster in this demo.



