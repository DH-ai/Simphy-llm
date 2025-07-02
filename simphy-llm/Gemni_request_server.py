from simphylib.splitter import PDFChunker
from simphylib.embedder import EmbeddingsSimphy
from simphylib.retriever import RetrieverSimphy
from simphylib.config import *
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
load_dotenv()
import logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

from typing import List, Union


import time
t = time.time()
def formater(text) -> str:
    """Format the text for better readability."""
    # Remove leading and trailing whitespace
    print(text)
    text = json.decoder.JSONDecoder().decode(text)
    print(text)
    
        # Replace multiple newlines with a single newline
    # text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text
    
from main_config import SCRIPT_DIR,SYSTEM_INSTRUCTION

## This is needed for chaining, or mememory, can save on disk and load later,
## also can retrive data in case of similar query, or can use it to train the model, maybe in a json format
contents_list = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""--- Result 1 ---
    2025-06-26 15:08:12,812 - INFO - Page: 22
    2025-06-26 15:08:12,812 - INFO - Content:
    NowaddfollowingcodetoonUpdate()propertyofworldscriptandsavescript(hopefullywithnocompilationerror).
    //get bodies as variablesvar disc=World.getBody(\"Disc\");var disc1=World.getBody(\"Disc1\");var disc2=World.getBody(\"Disc2\");//get body centersvar r1 = disc.getPosition();var r2 = disc1.getPosition();var r3 = disc2.getPosition();//get unit vectors joining centers of pair of discsvar r12 = (r1.to(r2)).getNormalized();var r23 = (r2.to(r3)).getNormalized();var r31 = (r3.to(r1)).getNormalized();//set velocity of each disc parallel to line joining itwith next discdisc.setVelocity(r12.x, r12.y);disc1.setVelocity(r23.x, r23.y);disc2.setVelocity(r31.x, r31.y);
    SimPhyScriptingGuide 142...
    2025-06-26 15:08:12,812 - INFO -

    --- Result 2 ---
    2025-06-26 15:08:12,812 - INFO - Page: 28
    2025-06-26 15:08:12,812 - INFO - Content:
    2. FromExistingObjects
    Clone of body can be created and added to world by calling functioncreateCopy(body)inworldobject.
    var d=World.getBody(\"Disc\");
    var d1=World.createCopy(d);
    //creates copy of Disc and adds to world
    Clones of body can also be created by calling copy()frombodyobject,but note that this function creates copy of object and returns it withoutaddingittoworld.
    This returned copy can later be added to world by usingworld.addBody();
    var d=World.getBody(\"Disc\");
    var d1=d.copy();
    World.addBody(d1);
    ChangingAppearance
    Thereexistfewfunctionsinbodyobjecttochangebody'scolor,imageandtext.
    var d=World.getBody(\"Disc\");
    d.setFillColor(“white”);
    d.setOutlineColor(new Color(\"red\"))
    d.setBrush(\"image1\");
    //make sure animation with name \"image1\" already exists
    d.setText(\"custom text\");...
    2025-06-26 15:08:12,812 - INFO -

    --- Result 3 ---
    2025-06-26 15:08:12,812 - INFO - Page: 27
    2025-06-26 15:08:12,812 - INFO - Content:
    CHAPTER4:
    BODIES&JOINTS
    AccessingBody/JointinScript
    Bodyor joint alreadyexistinginsimulationcanbeaccessedbycallinggetBody()andgetJoint()functionsof'world'object.
    Forexample:
    var d=World.getBody(\"Disc\");
    //stores body of name 'Disc' in variable d
    var jt=World.getJoint(\"joint1\");
    //stores joint of name 'joint1' in variable jt
    CreatingBody/Joints
    1. ByDefiningNewObjects
    NewBodyorjointcanbecreatedbycallingworld.addXXX()andworld.addXXXJoint()functions.
    Thefollowingcodecreatesarectangleandadiscconnectedbyaspringofforceconstant40N/m.
    var d = World.addDisc(0.5);
    var r = World.addRectangle(1, 1);
    r.setPosition(2, 0);
    var j = World.addSpringJoint(d, r, null, null, 30, 0);
    SimPhyScriptingGuide 147...

    Add a disc at origin of radius 2
    """),
            ],
        ),

        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Considering Disc Creation**

    I'm currently focused on incorporating the user's request to add a disc into the simulation.  I've found documentation detailing the `World.addDisc(radius)` function and how to set a body's position. My next step involves translating \"origin of radius 2\" into the appropriate function call, likely `World.addDisc(2)` and then setting the position to (0,0).


    **Synthesizing Code Implementation**

    My thought process is now focused on generating the final code. I've concluded that `World.addDisc(2);` is the most concise solution. The user's request, \"Add a disc at origin of radius 2\", simplifies to this single line, given the default origin and the explicit radius specification. I'm ready to output this code directly.


    """),
                types.Part.from_text(text="""```javascript
    World.addDisc(2);
    ```"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""--- Match 1 ---

    SimPhyScriptingGuide 135
    addDistanceJoint(Body body1, Body body2,
    Vector2 p1, Vector2 p2, double frequency,
    double damping)AddsDistanceJointbetweenbodiesbyconnectingthematcenters
    Joint
    addPolygon(Vector2[] vertices)Createspolygonal body
    Body
    addRectangle(double width, double height)Addsrectangleobjecttotheworld
    Body
    addRevoluteJoint(Body body1, Body body2,
    Vector2 v, double lowerLimit,
    double upperLimit)Addsrevolutejoint(hinge)
    Joint
    addSpringJoint(Body body1, Body body2,
    double SpringConstant)addsSpringjointbetweenbodiesbyconnectedtorespectivecenters
    Joint
    addTracer(Body body, Vector2 localPt)Addstrajectorytracertoabody
    Tracer
    clear()Clearsall Dynamicallygeneratedobjectsbyscript
    void
    createCopy(Body body)Createscopyofobjectandaddsittoworld
    Body

    --- Match 2 ---

    var d = World.getBody(\"Disc\");//don't let other forces interact with bodyd.setMass(0);//set initial position of bodyd.setPosition(0, 0);//create action objectvar a = Actions.moveTo(3, 0, 4, \"bounce\");//set action to bodyd.setAction(a);
    SimPhyScriptingGuide 150

    --- Match 3 ---

    CHAPTER4:
    BODIES&JOINTS
    AccessingBody/JointinScript
    Bodyor joint alreadyexistinginsimulationcanbeaccessedbycallinggetBody()andgetJoint()functionsof'world'object.
    Forexample:
    var d=World.getBody(\"Disc\");
    //stores body of name 'Disc' in variable d
    var jt=World.getJoint(\"joint1\");
    //stores joint of name 'joint1' in variable jt
    CreatingBody/Joints
    1. ByDefiningNewObjects
    NewBodyorjointcanbecreatedbycallingworld.addXXX()andworld.addXXXJoint()functions.
    Thefollowingcodecreatesarectangleandadiscconnectedbyaspringofforceconstant40N/m.
    var d = World.addDisc(0.5);
    var r = World.addRectangle(1, 1);
    r.setPosition(2, 0);
    var j = World.addSpringJoint(d, r, null, null, 30, 0);
    SimPhyScriptingGuide 147

    --- Match 4 ---

    ● Brush/Animation:loadedimageoranimation
    ● Sound:loadedsound
    ● Font :loadedfont
    ● File :loadedfile
    ● Action:whenattachedtobody,performssometask,oftenovertime
    Forexample:
    var d = World.addDisc(0.2);
    // create a new vector
    var pos = new Vector2(1, 0);
    // place body at 1,0
    d.setPosition(pos);
    // set color of body to red
    d.setFillColor(new Color(\"red\"));
    // find animation from resources folder with name \"image\"
    var brush = Resources.getAnimation(\"image\");
    // create copy of animation/brush and apply to body
    d.setBrush(brush.createCopy());
    //finally insert body in the world
    World.addBody(d);
    SimPhyScriptingGuide 122


    how to create a rectangle and attach spring?
    """),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Considering Spring Connections**

    I've been examining the documentation, specifically \"Match 3,\" to understand how to link shapes with a spring. The example uses a rectangle and a disc, connected by a spring. I see `World.addRectangle(1, 1);` creates a rectangle, and I'm figuring out the syntax for the spring connection.


    **Refining Joint Parameters**

    I'm now zeroing in on the `addSpringJoint` function and the crucial role of its parameters. It seems Match 1 uses a simplified version with just `SpringConstant`, but Match 3 introduces null values, indicating more complex adjustments are possible. Specifically, I'm focusing on how `addSpringJoint` works to connect the rectangle and disc. I am also investigating the `null` placeholders.


    **Analyzing Joint Placement**

    I'm now exploring the implications of the null values in `addSpringJoint`. It seems they might represent anchor points, possibly indicating where the spring connects on each body. If null, the connection defaults to the center of mass. I'm investigating how adjusting these anchor points would affect the spring's behavior and the overall visual outcome. I'm considering what would happen if the spring were connected to the corners of the shape.


    **Connecting with Springs**

    I've determined the core implementation is straightforward: create a rectangle, create a second body (like a disc), and then use `World.addSpringJoint` to connect them. The Match 3 example provides the most relevant code and I am ready to provide this code snippet.


    """),
                types.Part.from_text(text="""```javascript
    var r = World.addRectangle(1, 1);
    var d = World.addDisc(0.5);
    r.setPosition(2, 0);
    World.addSpringJoint(d, r, null, null, 30, 0);
    ```"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""/home/dhruv/Documents/Work/projects/simphy_NL_scripter/simphy-llm/rag_lama.py:28: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
    return self.retriever.get_relevant_documents(query)[:top_k]

    --- Match 1 ---

    setBodyPt(Vector2 bodyPt)Setstracepointonthebodyassociatedwithtracer
    void
    setColor(Color color)SetsFill Colorofthebody(todisablefill shapepassnull arguement)
    void
    setMode(int mode)SetsmodeusedtodrawTracercanbeSolid[0]ordotted[1]
    void
    setPtsCount(int count)Setsmaximumnumberofpointsintracerarray
    void
    setPtSize(int ptSize)Setspixel sizeoftracer
    void
    setShowAcceleration(boolean showAccele
    ration)Setsiflinearaccelerationistobedisplayedintracer
    void
    setShowAngularVelocity(boolean showAng
    lurVel)Setsifangularvelocityistobedispalyed
    void
    setShowVelocity(boolean showVelocity)Setsiflinearveclocityoftracerpointisdisplayed
    void
    SimPhyScriptingGuide 162

    --- Match 2 ---

    SimPhyScriptingGuide 135
    addDistanceJoint(Body body1, Body body2,
    Vector2 p1, Vector2 p2, double frequency,
    double damping)AddsDistanceJointbetweenbodiesbyconnectingthematcenters
    Joint
    addPolygon(Vector2[] vertices)Createspolygonal body
    Body
    addRectangle(double width, double height)Addsrectangleobjecttotheworld
    Body
    addRevoluteJoint(Body body1, Body body2,
    Vector2 v, double lowerLimit,
    double upperLimit)Addsrevolutejoint(hinge)
    Joint
    addSpringJoint(Body body1, Body body2,
    double SpringConstant)addsSpringjointbetweenbodiesbyconnectedtorespectivecenters
    Joint
    addTracer(Body body, Vector2 localPt)Addstrajectorytracertoabody
    Tracer
    clear()Clearsall Dynamicallygeneratedobjectsbyscript
    void
    createCopy(Body body)Createscopyofobjectandaddsittoworld
    Body

    --- Match 3 ---

    NowaddfollowingcodetoonUpdate()propertyofworldscriptandsavescript(hopefullywithnocompilationerror).
    //get bodies as variablesvar disc=World.getBody(\"Disc\");var disc1=World.getBody(\"Disc1\");var disc2=World.getBody(\"Disc2\");//get body centersvar r1 = disc.getPosition();var r2 = disc1.getPosition();var r3 = disc2.getPosition();//get unit vectors joining centers of pair of discsvar r12 = (r1.to(r2)).getNormalized();var r23 = (r2.to(r3)).getNormalized();var r31 = (r3.to(r1)).getNormalized();//set velocity of each disc parallel to line joining itwith next discdisc.setVelocity(r12.x, r12.y);disc1.setVelocity(r23.x, r23.y);disc2.setVelocity(r31.x, r31.y);
    SimPhyScriptingGuide 142

    --- Match 4 ---

    var d = World.getBody(\"Disc\");//don't let other forces interact with bodyd.setMass(0);//set initial position of bodyd.setPosition(0, 0);//create action objectvar a = Actions.moveTo(3, 0, 4, \"bounce\");//set action to bodyd.setAction(a);
    SimPhyScriptingGuide 150



    Create a triangular wedge with sides 6, 8 and 10 and mass 5
    """),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Define the Object**

    I've identified the object type: a triangular wedge, which translates to a polygon in SimPhy.  Currently, I'm poring over the documentation to locate the suitable function for polygon creation. The `addPolygon()` function seems promising. I'm now exploring how to accurately define the vertices.


    **Assembling the Script**

    I'm now in the process of assembling the SimPhy script.  The core logic revolves around defining vertices, creating the polygon, setting the mass, and ensuring the correct side lengths (6, 8, and 10). I'm constructing the code step by step, ready to be tested and refined. The vertices will be defined at (0,0), (8,0), and (0,6), to create a right-angled triangle. I'll test it out and see how it goes.


    **Crafting the SimPhy Script**

    I've assembled the SimPhy script, focusing on the core steps: defining vertices, creating the polygon via `addPolygon()`, and setting the mass to 5 using `setMass()`. The vertices are at (0, 0), (8, 0), and (0, 6) to construct the desired triangle. Now, it's time to test the script to ensure the correct geometric representation and mass properties.
    ```javascript
    // Step 1: Define  vertices for a 6, 8, 10 triangle.
    // Let's use (0,0), ( 8,0), and (0,6). The sides will be:
    // - distance between (0,0) and ( 8,0) is 8.
    // - distance between (0,0) and (0,6)  is 6.
    // - distance between (8,0) and (0,6) is sqrt((8-0)^ 2 + (0-6)^2) = sqrt(64 + 36) = sqrt(10 0) = 10.
    // This works.
    var vertices = [
    new Vector2(0,  0),
    new Vector2(8, 0),
    new Vector2(0, 6 )
    ];

    // Step 2: Create the polygon.
    var wedge = World.addPolygon(vertices);

    // Step 3: Set the mass.
    wedge.setMass(5);
    ```


    """),
                types.Part.from_text(text="""```javascript
    var vertices = [new Vector2(0, 0), new Vector2(8, 0), new Vector2(0, 6)];
    var wedge = World.addPolygon(vertices);
    wedge.setMass(5);
    ```"""),
            ],
        ),
    ]


def generate(content):
    """Generate content using the Gemini API."""
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = GENMODEL
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=SYSTEM_INSTRUCTION),
        ],
        temperature=0,
        candidate_count=1,
        # response_mime_type="application/json",
        top_p=0.63,
        top_k=20,
        seed=1,
        max_output_tokens=300,
        # stop_sequences=["STOP!"],
        # presence_penalty=0.0,
        frequency_penalty=1.2,
        # num_beams=1,
    )

    # response = client.models.generate_content(
    #     model=model,
    #     contents=content,
    #     config=generate_content_config,
    # )
    # formater(response.text)
    # print(response.text)
    model_output_temp=""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=content,
        config=generate_content_config,
    ):
        model_output_temp = model_output_temp +"\n" +str(chunk.text)
        
    # resposne = formater(model_output_temp)
    print(model_output_temp)
    new_model_content = types.Content(
        role="model",
        parts=[types.Part.from_text(text= model_output_temp)]
    )

    
    
    return new_model_content


def output_results(docs, query=None):
    """Format and print the retrieved documents."""
    assert isinstance(docs, list), "docs should be a list of Document objects"
    assert query is None or isinstance(query, str), "query should be a string or None"
    

    rag_result = "\n".join([f"Content:{doc.page_content}"for doc in docs])
    # print(rag_result) -> see for if there are spaces in betweeen
    new_user_content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=f"Rag_result: {rag_result} \n\n Query: {query}") 
        ],
    )
    # print(new_user_content.parts)
    new_model_content = generate(new_user_content)

    # logging.info(f"Result of the query: {query}\n\n".format(query=query))
    # if not new_model_content:print("-"*17 + "\n" +"{t}".format(t = new_model_content.parts[0]) +"\n"+"-"*17 )
    # logging.info("\n\nEnd of SLiPI Outout ")
    


 

if __name__ == "__main__":
    


    
    logging.info("This is SLiPI, your SimPhy Scripting Assistant.")
    logging.info("Loading PDF and creating vector store...")
    # if not PDFChunker().check_vectorstore_before_load():
    pdf_chunker = PDFChunker(pdf_path=SCRIPT_DIR+"/docs/SimpScriptG.pdf", chunk_size=1000, chunk_overlap=100)
    pdf_chunker.load()
    chunks = pdf_chunker.split()
    # chunks = pdf_chunker.format_chunks()
    embedder = EmbeddingsSimphy(model_name=HUGGINGFACE_EMBEDDING_MODEL_BAAI,save_vectorstore=False)
    vectorstore = embedder.create_vectorstore(chunks)
    # else:
    #     logging.info("Vector store already exists. Loading from cache...")
    #     embedder = EmbeddingsSimphy(model_name=HUGGINGFACE_EMBEDDING_MODEL_BAAI)
    #     vectorstore = embedder.load_vectorstore()
    if vectorstore is None:
        logging.error("Failed to load vector store. Exiting.")
        exit(1)
    retriever = RetrieverSimphy(vectorstore=vectorstore)
    

    logging.info("Enter your queries below. Type 'quit', 'exit', or 'q' to end the session.")
    print(time.time() - t)
    while True:
        query = input("Query: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query.lower() in ['clear', 'cls']:
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        if not query.strip():
            logging.warning("Please enter a valid query.")
            continue

        
        
        doc = retriever.retrieve(query=query, k=7)
    
        logging.info(f"Result of the query: {query}\n\n".format(query=query))
        # for i, doc2 in enumerate(doc, 1):
                
        #         print(f"\n\n--- Result {i} ---\n\n")
        #         print(f"Page: {doc2.metadata.get('page', 'Unknown')}")
        #         print(f"Content: \n{doc2.page_content}...")
        # logging.info("\n\n End of RAG Outout ")
        output_results(doc,query)

            

