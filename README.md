# pymap_meme
Little python code to generate and plot the polynomial that crosses the biggest N cities of a given country via lagrange interpolation

This started as a meme, on which some users where posting statements like "The three biggest cities on Germany fits perfectly on a circle", although it's always possible to create a circle with three non-collinear points. Later versions where made so that the two biggest cities fits on a line or the four biggest cities fits perfectly on a third degree polynomial (you get the idea). This code just generalize this idea. 

It works (kinda), the only problem is that it fails for a big number of cities, with that "big" varying depending on the country

I didn't tried to solve that because the concept itself has a fatal fail, it may not be possible to plot it at all because the points of the cities coordinates could not compose a function (Two cities could have the same x value). 

Also, the exponents of the function may rise very quickly (It's expected from the lagrange interpolation), resulting in a function that looks like vertical lines on the map.

With all that said, here's how to use it:

### Usage
It's recommended to use a venv before installing the requirements
```sh
python -m venv venv
source venv/bin/activate
```
Install the requirements
```sh
pip install -r requirements.txt
```
With that, you can simply run the code with
```sh
python city_map.py
```
and follow the instructions.