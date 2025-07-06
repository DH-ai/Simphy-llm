1. Prompt : "Write the code to create a radial gradient that transitions from yellow to red and apply it as the fill style for a circle."
    
    Code By me:
    ```javascript
    var canvas = Widgets.getCanvas("canvas");                                                                                                                                                        
    var context = canvas.getContext                                                                                                                                                                  
    ();                                                                                                                                                                                              
    var rg = context.createRadialGradient(80, 80, 20, 120, 120, 110);                                                                                                                                
    rg.addColorStop(0, 'yellow');                                                                                                                                                                    
    rg.addColorStop(1, 'red');                                                                                                                                                                       
    context.fillStyle = rg;                                                                                                                                                                          
                                                                                                                                                                                                    
    context.beginPath();                                                                                                                                                                             
    context.arc(120, 120, 110, 0, 2 * Math.PI, false);                                                                                                                                               
    context.closePath();                                                                                                                                                                             
    context.fill();                                                                                                                                                                                  
    ```

    Google API output:

    ```javascript
    
    var canvas = Widgets.getCanvas("canvas");
    var context = canvas.getContext(); 1  1 

    // 1. Create a radial gradient. 1  1  1 
    // The arguments define a start circle (x0,y0,r0) and an end circle (x1,y1,r1). 1 
    var rg = context.createRadialGradient(80, 80, 20, 120, 120, 110); 1  1 

    // 2. 1  Add color stops to the gradient. 1  1  1  1 
    rg.addColorStop(0, 'yellow');
    rg.addColorStop(1, 'red'); 1  1 

    // 3. 1  Set the fill style to the new gradient. 1  1  1 
    context.fillStyle = rg; 1  1  1 

    // 4. 1  Draw a filled circle. 1  1  1 
    context.beginPath();
    context.arc(120, 120, 110, 0, 2 * Math.PI, false);
    context.closePath();
    context.fill();
    ```

2. Prompt: 

    code_me : How do you create a prompt box that asks a user to input their age?

    ```javascript
    var age = App.readInteger("Enter Your Age. 1  Don't worry ladies, you can tell me lies! ")
    ```

    google_
    ```javascript
    var age = App.readInteger("Enter Your Age. 1  Don't worry ladies, you can tell me lies! ")
    ```
3. Prompt: Create a disc with initial velocity (10, 5) under gravity. Plot position vs time.

    code:
    ```javascript
     function world_onStart() {                                                                                        
    // Set world gravity (e.g., Earth's gravity)                                                                    
                                                                                                                    
    World.setGravity(0, -9.8);                                                                                      
                                                                                                                    
    // Create a new disc and name it                                                                                
    var d = World.addDisc(0.5); // Creates a disc with radius 0.5                                                   
    d.setName("Projectile");                                                                                        
                                                                                                                    
    // Set the initial velocity of the disc to                                                                      
    (10, 5) m/s                                                                                                      
    d.setVelocity(10, 5);                                                                                           
                                                                                                                    
    // Attach a tracer to the center of the disc to plot its position                                               
    var tr = World.addTracer(d, null);                                                                              
                                                                                                                    
    // Optional: Customize the tracer's appearance                                                                  
                                                                                                                    
    tr.setColor(new Color("red"));                                                                                  
    }   
    ```


    google_
    ```javascript
    // Create a new disc and store it in a variable
    var disc = World.addDisc(0.5); 1 

    // Set the initial velocity of the disc to (10, 5)
    disc.setVelocity(10, 5); 1 

    // Add a tracer to the disc's center to plot its trajectory
    World.addTracer(disc, null);
    ```
    
4. Prompt: What is the difference between the readChoice and readChoices methods?

    code:
    ```javascript
                                                                                                                                                                                                        
    const choices = ["Option A", "Option B", "                                                                                                                                                       
    Option C", "Option D"];                                                                                                                                                                          
                                                                                                                                                                                                    
    const message1 = "readChoice: Select exactly one option. Returns a single integer index.";                                                                                                       
    const singleChoiceIndex = App.readChoice(message1, choices);                                                                                                                                     
    App.alert("readChoice returned: " + singleChoiceIndex);                                                                                                                                          
                                                                                                                                                                                                    
    const message2 = "                                                                                                                                                                               
    readChoices: Select zero or more options. Returns an array of integer indices.";                                                                                                                 
    const multipleChoiceIndices = App.readChoices(message2, choices);                                                                                                                                
    App.alert("readChoices returned: " + multipleChoiceIndices.toString());     
    ```

    google:

    Based on the provided documents, here are the differences between the readChoice and readChoices methods in SimPhy:

    - readChoice: This method displays a box with choices as radio buttons, allowing a user to select exactly one option. 1
    - readChoices: This method displays a box with choices as checkboxes, allowing a user to select none, one, or multiple options. 1 

5. Prompt: Write the code to draw a red line that starts at (30, 90), curves in an arc, and then continues straight to (390, 90)

    code:
    ```javascript
    var canvas = Widgets.getCanvas("canvas");
    var context = canvas.getContext(); 
    context.beginPath();                                                                                                                                                                             
    context.strokeStyle = 'red';                                                                                                                                                                     
                                                                                                                                                                                                    
    context.lineWidth = 2;                                                                                                                                                                           
    context.moveTo(30, 90);                                                                                                                                                                          
    context.arcTo(210, 30, 390, 90, 80);                                                                                                                                                             
    context.lineTo(390, 90);                                                                                                                                                                         
    context.stroke();
    ```
    
    Google_
    ```javascript
    var canvas = Widgets.getCanvas("canvas");
    var context = canvas.getContext(); 

    // 1. Start a new path.
    context.beginPath(); 

    // 2.  Move to the starting point (30, 90). 
    context.moveTo(30, 90); 

    // 3.  Draw a curve that ends at (270, 90). 
    context.quadraticCurveTo(150, 150, 270, 90);

    // 4. Draw a straight line to the end point (390, 90). 
    context.lineTo(390, 90); 

    // 5. 1  Set the line color to red. 
    context.strokeStyle = "red"; 

    // 6. Render the line on the canvas. 
    context.stroke();
    ```

6. Prompt: "Create a blue rectangle (2m x 3m) with 0.5 opacity at position (5,10)"

    code: 
    ```javascript
    const canvas = Widgets.getCanvas("canvas");                                                                                                                                                      
    const                                                                                                                                                                                            
    context = canvas.getContext("2d");                                                                                                                                                              
    context.fillStyle = "rgba(0, 0, 255, 0.5)";                                                                                                                                                      
    context.fillRect(5, 10, 2, 3);    
    ```

    google_
    ```javascript

    context.globalAlpha = "0.5";
    context.fillStyle = 'blue';
    context.fillRect(5, 10, 2, 3);
    ```

7. Prompt: Connect discA and discB with a spring joint (k=75 N/m) and set joint color to green

    code:
    ```javascript
    var discA = World.getBody("discA");                                                                                                                                                              
    var discB = World.getBody("discB");                                                                                                                                                              
                                                                                                                                                                                                    
    if (discA                                                                                                                                                                                        
    && discB) {                                                                                                                                                                                     
        var spring = World.addSpringJoint(discA, discB, null, null, 75, 0);                                                                                                                          
        spring.setColor(new Color("green"));                                                                                                                                                         
    }  
    ```

    google
    ```javascript
    var discA = World.getBody("discA");
    var discB = World.getBody("discB");
    var j = World.addSpringJoint(discA, discB, 75);
    j.setColor(new Color("green"));
    ```
8. Prompt: "Create 10 discs in a circle with radius 5m. Make odd-numbered discs twice as massive as even ones"

    code:

    ```javascript
    function world_onStart() {                                                                                                                                                                       
        const numDiscs = 10;                                                                                                                                                                         
        const circleRadius = 5;                                                                                                                                                                      
        const baseMass = 1;                                                                                                                                                                          
        const discBodyRadius = 0                                                                                                                                                                     
    .4;                                                                                                                                                                                              
                                                                                                                                                                                                    
        for (let i = 0; i < numDiscs; i++) {                                                                                                                                                         
            const angle = (i / numDiscs) * 2 * Math.PI;                                                                                                                                              
            const x = circleRadius * Math.cos(angle);                                                                                                                                                
            const y = circleRadius *                                                                                                                                                                 
    Math.sin(angle);                                                                                                                                                                                
                                                                                                                                                                                                    
            const disc = World.addDisc(discBodyRadius);                                                                                                                                              
            disc.setPosition(x, y);                                                                                                                                                                  
                                                                                                                                                                                                    
            if ((i + 1) % 2 !== 0) {                                                                                                                                                                 
                disc.setMass(baseMass * 2);                                                                                                                                                          
            } else                                                                                                                                                                                   
    {                                                                                                                                                                                               
                disc.setMass(baseMass);                                                                                                                                                              
            }                                                                                                                                                                                        
        }                                                                                                                                                                                            
    }                                                                                                                                                                                                
        
    ```

    Google_

    ```javascript

    var discArray = [];
    var numDiscs = 10;
    var radius = 5;

    for (var i = 0; i < numDiscs; i++) {
        var d = World.addDisc(0.4); // Assuming a disc radius of 0.4m

        // Calculate position on a circle
        var angle = i * 2 * Math.PI / numDiscs;
        var x = radius * Math.cos(angle);
        var y = radius * Math.sin(angle);
        d.translate(x, y);

        // Set mass for odd and even discs
        if (i % 2 == 0) { // Even disc
            d.setMass(1);
        } else { // Odd disc
            d.setMass(2);
        }

        discArray.push(d);

    ```

9. Prompt :When mouse clicks on a disc named 'target', make it apply 50N force upward"

    code
    ```javascript
                                                                                                                                                                                                      
    World.clear();                                                                                                                                                                                   
                                                                                                                                                                                                    
    var ground = World.addRectangle(10, 0.5);                                                                                                                                                        
    ground.setStatic(true);                                                                                                                                                                          
    ground.setPosition(0, -4);                                                                                                                                                                       
                                                                                                                                                                                                    
                                                                                                                                                                                                    
    var target = World.addDisc(0.5);                                                                                                                                                                 
    target.setName("target");                                                                                                                                                                        
    target.setPosition(0, 0);                                                                                                                                                                        
    target.setMass(1);                                                                                                                                                                               
                                                                                                                                                                                                    
    function world_onMouseEvent(x, y, worldPt, clickCount, id, button) {                                                                                                                             
        var body                                                                                                                                                                                     
    = World.getBodyAt(worldPt);                                                                                                                                                                     
        if (body != null && body.getName() == "target") {                                                                                                                                            
            body.applyForce(0, 50);                                                                                                                                                                  
        }                                                                                                                                                                                            
    } 
    ```

    google 

    ```javascript
    function world_onMouseEvent(x, y, worldPt, clickCount, id, button) {
    // Get the body at the mouse click location
        var clickedBody = World.getBodyAt(worldPt);
        
        // Get the specific disc named "target"
        var targetDisc = World.getBody("target");

        // Check if the clicked body is the target disc
        if (clickedBody != null && clickedBody == targetDisc) {
            // Apply 50N force upward (in the positive y-direction)
            clickedBody.applyForce(0, 50);
        }
    }
    ```

10. prompt: "Make a disc fade out over 2 seconds, then move to (8,4) with bounce easing"

    code
    ```javascript
     function world_onStart() {                                                                                                                                                                       
        var disc = World.add Disc(0.5);                                                                                                                                                                                       
        disc.setPosition(-8, 0);                                                                                                                                                                     
        disc.setRestitution(0.5);                                                                                                                                                                    
        disc.setFillColor("#3498db");                                                                                                                                                                
                                                                                                                                                                                                    
        var fadeAction = Actions.fadeOut(2);                                                                                                                                                         
        var moveAction = Actions.moveTo(8,                                                                                                                                                           
    4, 2, Actions.bounceIn);                                                                                                                                                                         
        var sequence = Actions.sequence(fadeAction, moveAction);                                                                                                                                     
                                                                                                                                                                                                    
        disc.setAction(sequence);                                                                                                                                                                    
    }      
    ```

    google 
    ```javascript
    // Get a reference to the disc you want to animate.
    // Replace "myDisc" with the actual name of your disc.
    var disc = World.getBody("myDisc");

    // Create a sequence of actions. 1  1  1 
    var sequence = Actions.sequence(
        // First, fade the disc out over a duration of 2 seconds.
        Actions.fadeOut(2),

        // After the fade is complete, move the disc to position (8, 4).
        // The move will take 1.5 seconds and use "bounce" interpolation.
        Actions.moveTo(8, 4, 1.5, "bounce")
    );

    // Apply the action sequence to the disc.
    disc.setAction(sequence);
    ```

11. Prompt: "Display velocity magnitude on a disc in red bold text, right-aligned at local (0.5, -0.5)"

    code:
    ```javascript
     var disc = World.getBody("Disc");                                                                                                                                                                
                                                                                                                                                                                                  
    if (disc) {                                                                                                                                                                                      
        Sim.onTick = function() {                                                                                                                                                                    
            var vel = disc.getVelocity();                                                                                                                                                            
            var mag = vel.getMagnitude();                                                                                                                                                            
            disc.text = '[font="default-large" color="                                                                                                                                               
    red" xalign="2" x="0.5" y="-0.5"]' + mag.toFixed(2);                                                                                                                                             
        };                                                                                                                                                                                           
    }                                                                                                                                                                                                
        
    ```

    google
    ```javascript
    // Get a reference to the disc. 1  Replace "myDisc" with the actual name of your disc.
    var disc = World.getBody("myDisc");

    // Add a tracer to the disc. 1 
    var tr = World.addTracer(disc); 1 

    // Set the tracer's position to the local coordinates (0.5, -0.5) on the disc.
    tr.setBodyPt(new Vector2(0.5, -0.5));

    // Enable the velocity display for the tracer.
    tr.setShowVelocity(true);

    // Set the color of the tracer and its text to red.
    tr.setColor(new Color("red")); 1 

    // Note: The provided documentation does not specify a method
    // to make the tracer's text bold or to change its alignment.
    ```
12. Prompt: "Create 5 rotated copies (45° each) of polygon 'template' around point (0,0)"

    code:
    ```javascript
     const template = World.getBody("template");                                                                                                                                                      
                                                                                                                                                                                                    
    if (template) {                                                                                                                                                                                  
        for (let i = 1; i <= 5; i++) {                                                                                                                                                               
                                                                                                                                                                                                    
            const copy = template.clone();                                                                                                                                                           
            copy.name = `rotated_copy_${i}`;                                                                                                                                                         
            World.add(copy);                                                                                                                                                                         
            const angle = i * 45 * Math.PI / 180;                                                                                                                                                    
            const center = new Vector(0, 0);                                                                                                                                                         
                                                                                                                                                                                                    
    copy.rotate(angle, center);                                                                                                                                                                      
        }                                                                                                                                                                                            
        template.setVisible(false);                                                                                                                                                                  
    }      
    ```

    google

    ```javascript
    // Get the original polygon object. 1 
    var template = World.getBody("template"); 1 

    // Define the rotation angle in radians. 1 
    var angleInRadians = 45 * Math.PI / 180; 1 

    // Loop 5 times to create rotated copies. 1 
    for (var i = 0; i < 5; i++) {
        // The rotate() function modifies the drawing context for subsequent operations. 1 
        // Rotations are cumulative. 1 
        context.rotate(angleInRadians); 1  1 

        // Create a copy of the template body. 1 
        World.createCopy(template);
    } 1 

    // Optional: Reset the transformation to avoid affecting subsequent drawings.
    context.setTransform(1, 0, 0, 1, 0, 0);
    ```

    Trash responses, deep seek fried both of them

    ```javascript
    // Get the template polygon body
    var template = World.getBody("template");

    // Define rotation parameters
    var center = new Vector2(0, 0);     // Center point
    var radius = 5;                     // Distance from center
    var copies = 5;                     // Number of copies
    var angleStep = 45 * (Math.PI/180); // 45° in radians

    // Create rotated copies
    for (var i = 0; i < copies; i++) {
        // Calculate angle for this copy
        var angle = i * angleStep;
        
        // Create copy
        var copy = World.createCopy(template);
        
        // Calculate position
        var x = center.x + radius * Math.cos(angle);
        var y = center.y + radius * Math.sin(angle);
        
        // Position and rotate
        copy.setPosition(x, y);
        copy.setRotation(angle);
        
        // Optional: Differentiate copies visually
        copy.setFillColor(Color.HSVtoRGB(i/5.0, 1, 1));
    }
    ```

