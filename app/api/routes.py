from fastapi import APIRouter,HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner
import asyncio
from enum import Enum
from agents.guardrail import GuardrailFunctionOutput,input_guardrail

router = APIRouter(prefix="/travel", tags=["Travel"])

class BudgetType(str, Enum):
    budget = "budget"
    mid_range = "mid-range"
    luxury = "luxury"

class BudgetBreakdown(BaseModel):
    accommodation: int | str
    food: int | str
    activities: int | str
    transportation: int | str

class BudgetOption(BaseModel):
    budget_type: BudgetType
    estimate: int | str
    breakdown: BudgetBreakdown

class ItineraryInput(BaseModel):
    location: str
    start_date: str
    destination: str
    duration: int

class TravelPlannerDay(BaseModel):
    date: str
    activities: list[str]

class TravelPlannerOutput(BaseModel):
    title: str
    itinerary: list[TravelPlannerDay]
    budget: list[BudgetOption]

class ValidPlaceOutput(BaseModel):
    is_valid_place: bool

guardrail_agent = Agent( 
    name="Place validation",
    instructions="Check if the user is specifying a valid from and to location for travel.",
    output_type=ValidPlaceOutput,
    model="gpt-4o-mini"
)

@input_guardrail
async def guardrail_valid_places(ctx, agent, input_data):
    
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    is_valid_place = result.final_output.is_valid_place
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered= not is_valid_place)


load_dotenv(override=True)



@router.post("/itinerary",response_model = TravelPlannerOutput)
async def  generate_itinerary(request: ItineraryInput):
    try:
        instructions = """
You are a travel planner agent. Your task is to create a travel itinerary for the given destination and duration.
The itinerary should include daily activities, places to visit, and any necessary travel arrangements. Also give asuitable user attractive title to the itinerary. 
Also, suggest 3 different types of budget options in Indian Rupees for the itinerary - budget, mid-range and luxury. Each option should include the estimated cost for the entire trip and a breakdown of costs for accommodation, food, activities, and transportation.
""";
        input = f"Plan a trip to {request.destination} from {request.location},starting on {request.start_date} for a duration of {request.duration}."
        travel_agent = Agent(
        name="Travel Planner Agent",
        instructions=instructions,  
        model = "gpt-5.4",
        output_type= TravelPlannerOutput,
        input_guardrails = [guardrail_valid_places])
        result  = await Runner.run(travel_agent,input)
        print(result.final_output)
        return TravelPlannerOutput.model_validate(result.final_output)
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    


