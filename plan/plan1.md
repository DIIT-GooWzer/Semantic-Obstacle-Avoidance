From what we discussed:

Change from hardcode categories to stress level categories

3 Stress categories should be defined - Low, Medium, High

Stress levels should not directly jump into the value, have to change the logic where it increases the value by accumalating smaller numbers like 0.05 till it reaches the value - this is to simulate the realistic stress hormone we have to the robot, where stess levels are not immediately risen or lowered, it slowly reaches that level depends on the active feedback of observasations and response

Additionally, I have to develop a GUI where a graph will be generated that shows on the x axis the stress level and y axis the potential, where the it will display the robot's current stress level while oscilating back and forth depending on the situation. 

Potential is defined as the amount of action the robot is about to take relative to the amount of stress it is experiecing

Stress levels can be defined from 2 or more things, but in this case i can only think of two:
- The size of the obstacle - bigger size means the obstacle is really close thus higher stress level
- Obstacle identified - biological or stationary non living objects? And in biological if its aggresive or passive? These should be defined from the LLM's perception and decide whether it is a threat and coughs up a value 

From what we have discussed we will be using this formula:
Stress Level = Size of the image + LLM perception of threat //there can be better formula than this, this is only what i am currently thinking off right now
 
Accumaltion Constant x Stress Level =  Potential

Accumulation Constant can be 0.01

Then in this case here are some important questions to think about
- On what point should the value of stress levels should be taken and used during the accumalation to define the robot's set of actions - should it have a 5 - 7 seconds timer on the accumation process?
- How can we balance the rate of change in the stress level so that it doesn't go too slow or too fast

Different Stages of Stress and their course of action supposingly
Low level
Just standard avoid/bypass, basicailly what we have done now

Medium Level
Slightly Alert, taking small steps and scanning every step (prolly what we've done for like each prompt step and stuff), pauses more to detect distance whether it goes away or approaches the robot


High Level
Hight alert, taking active steps to backoff from the obstacle, skipping to another lane if taken too long (while lane shift it will take bigger edge steps trying to completely avoid the threat) - triggered either by LLM threat perception, or obstacle is actively moving towards the robot

Anyways this is so far the planning mindmap, what are the questions do you have?
