"""
Example Agent Implementation
Demonstrates how to create a simple agent using the base agent framework
"""

from typing import Any
from agents.core.base_agent import BaseAgent, AgentContext, AgentResult


class EchoAgent(BaseAgent):
    """
    Simple echo agent that returns the input with some processing
    Demonstrates the basic agent structure
    """
    
    def __init__(self, config: dict = None):
        super().__init__(
            name="echo-agent",
            version="1.0.0",
            description="Simple agent that echoes input with additional information",
            capabilities=["text_processing", "echo", "formatting"],
            config=config
        )
    
    async def process(self, input_data: Any) -> Any:
        """
        Main processing logic for the echo agent
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed echo response
        """
        if input_data is None:
            return {"error": "No input provided", "echo": None}
        
        # Simple echo processing
        if isinstance(input_data, str):
            processed_response = {
                "original_input": input_data,
                "echo": f"Echo: {input_data}",
                "length": len(input_data),
                "word_count": len(input_data.split()),
                "processed_by": self.metadata.name,
                "agent_version": self.metadata.version
            }
        elif isinstance(input_data, dict):
            processed_response = {
                "original_input": input_data,
                "echo": f"Received dictionary with {len(input_data)} keys",
                "keys": list(input_data.keys()),
                "processed_by": self.metadata.name,
                "agent_version": self.metadata.version
            }
        elif isinstance(input_data, (list, tuple)):
            processed_response = {
                "original_input": input_data,
                "echo": f"Received sequence with {len(input_data)} items",
                "item_count": len(input_data),
                "processed_by": self.metadata.name,
                "agent_version": self.metadata.version
            }
        else:
            processed_response = {
                "original_input": str(input_data),
                "echo": f"Received {type(input_data).__name__}: {str(input_data)}",
                "type": type(input_data).__name__,
                "processed_by": self.metadata.name,
                "agent_version": self.metadata.version
            }
        
        return processed_response


class TextSummarizerAgent(BaseAgent):
    """
    Text summarizer agent that provides basic text analysis
    More sophisticated than echo agent but still simple
    """
    
    def __init__(self, config: dict = None):
        super().__init__(
            name="text-summarizer",
            version="1.0.0",
            description="Agent that provides basic text analysis and summarization",
            capabilities=["text_analysis", "summarization", "word_count", "statistics"],
            config=config
        )
    
    async def process(self, input_data: Any) -> Any:
        """
        Process text input and provide analysis
        
        Args:
            input_data: Text input to analyze
            
        Returns:
            Text analysis results
        """
        if not isinstance(input_data, str):
            return {
                "error": "Text summarizer only processes string input",
                "received_type": type(input_data).__name__
            }
        
        if not input_data.strip():
            return {
                "error": "Empty text provided",
                "analysis": None
            }
        
        # Basic text analysis
        text = input_data.strip()
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Character analysis
        chars_total = len(text)
        chars_no_spaces = len(text.replace(' ', ''))
        
        # Word analysis
        unique_words = set(word.lower().strip('.,!?;:"()[]{}') for word in words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simple readability score (Flesch-like approximation)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words) if words else 0
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        # Generate simple summary (first sentence + last sentence if more than 2 sentences)
        summary = sentences[0] if sentences else ""
        if len(sentences) > 2:
            summary += f" ... {sentences[-1]}"
        
        analysis_result = {
            "input_text": text,
            "summary": summary,
            "statistics": {
                "character_count": chars_total,
                "character_count_no_spaces": chars_no_spaces,
                "word_count": len(words),
                "unique_word_count": len(unique_words),
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "average_word_length": round(avg_word_length, 2),
                "average_sentence_length": round(avg_sentence_length, 2),
                "estimated_reading_time_minutes": round(len(words) / 200, 1)  # ~200 WPM
            },
            "readability": {
                "score": round(readability_score, 1),
                "level": self._get_readability_level(readability_score)
            },
            "processed_by": self.metadata.name,
            "agent_version": self.metadata.version
        }
        
        return analysis_result
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter"""
        word = word.lower().strip('.,!?;:"()[]{}')
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)  # Every word has at least 1 syllable
    
    def _get_readability_level(self, score: float) -> str:
        """Convert Flesch score to readability level"""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate that input is text"""
        return isinstance(input_data, str) and bool(input_data.strip())


class MathAgent(BaseAgent):
    """
    Math agent that can perform basic mathematical operations
    Demonstrates more complex input validation and error handling
    """
    
    def __init__(self, config: dict = None):
        super().__init__(
            name="math-agent",
            version="1.0.0",
            description="Agent that performs basic mathematical operations",
            capabilities=["arithmetic", "basic_math", "calculations"],
            config=config
        )
    
    async def process(self, input_data: Any) -> Any:
        """
        Process mathematical operations
        
        Expected input format:
        {
            "operation": "add|subtract|multiply|divide|power|sqrt",
            "operands": [number1, number2, ...] or number (for unary operations)
        }
        or
        "expression": "mathematical expression as string"
        
        Args:
            input_data: Math operation specification
            
        Returns:
            Mathematical result
        """
        if isinstance(input_data, str):
            return await self._process_expression(input_data)
        elif isinstance(input_data, dict):
            return await self._process_operation(input_data)
        else:
            return {
                "error": "Invalid input format. Expected string expression or operation dict",
                "received_type": type(input_data).__name__
            }
    
    async def _process_expression(self, expression: str) -> dict:
        """Process a mathematical expression string"""
        try:
            # Simple expression evaluation (unsafe in production!)
            # In production, you'd use a proper math parser
            result = eval(expression)
            
            return {
                "expression": expression,
                "result": result,
                "type": "expression_evaluation",
                "processed_by": self.metadata.name
            }
            
        except ZeroDivisionError:
            return {
                "error": "Division by zero",
                "expression": expression,
                "type": "mathematical_error"
            }
        except Exception as e:
            return {
                "error": f"Invalid mathematical expression: {str(e)}",
                "expression": expression,
                "type": "parsing_error"
            }
    
    async def _process_operation(self, operation_data: dict) -> dict:
        """Process a structured mathematical operation"""
        try:
            operation = operation_data.get("operation", "").lower()
            operands = operation_data.get("operands", [])
            
            if not operation:
                return {"error": "No operation specified"}
            
            if operation in ["add", "sum"]:
                result = sum(operands)
            elif operation in ["subtract", "sub"]:
                if len(operands) < 2:
                    return {"error": "Subtraction requires at least 2 operands"}
                result = operands[0] - sum(operands[1:])
            elif operation in ["multiply", "mul"]:
                result = 1
                for num in operands:
                    result *= num
            elif operation in ["divide", "div"]:
                if len(operands) != 2:
                    return {"error": "Division requires exactly 2 operands"}
                if operands[1] == 0:
                    return {"error": "Division by zero"}
                result = operands[0] / operands[1]
            elif operation in ["power", "pow"]:
                if len(operands) != 2:
                    return {"error": "Power operation requires exactly 2 operands"}
                result = operands[0] ** operands[1]
            elif operation in ["sqrt", "square_root"]:
                if len(operands) != 1:
                    return {"error": "Square root requires exactly 1 operand"}
                if operands[0] < 0:
                    return {"error": "Cannot calculate square root of negative number"}
                result = operands[0] ** 0.5
            else:
                return {"error": f"Unknown operation: {operation}"}
            
            return {
                "operation": operation,
                "operands": operands,
                "result": result,
                "type": "mathematical_operation",
                "processed_by": self.metadata.name
            }
            
        except Exception as e:
            return {
                "error": f"Error processing mathematical operation: {str(e)}",
                "operation_data": operation_data,
                "type": "processing_error"
            }
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate mathematical input"""
        if isinstance(input_data, str):
            # Basic validation for string expressions
            return bool(input_data.strip())
        elif isinstance(input_data, dict):
            # Validate operation dict structure
            if "operation" not in input_data:
                return False
            operands = input_data.get("operands", [])
            if not isinstance(operands, list):
                return False
            # Check that all operands are numbers
            return all(isinstance(x, (int, float)) for x in operands)
        return False


# Example usage and testing functions
async def demo_agents():
    """Demonstrate the example agents"""
    print("=== AI Agent Orchestration Platform - Agent Demo ===\n")
    
    # Echo Agent Demo
    print("1. Echo Agent Demo")
    echo_agent = EchoAgent()
    
    test_inputs = [
        "Hello, World!",
        {"name": "John", "age": 30, "city": "New York"},
        [1, 2, 3, 4, 5],
        42
    ]
    
    for i, input_data in enumerate(test_inputs, 1):
        print(f"\nTest {i}: {input_data}")
        result = await echo_agent.execute(input_data)
        print(f"Result: {result.data}")
        print(f"Success: {result.success}, Confidence: {result.confidence}")
    
    # Text Summarizer Demo
    print("\n\n2. Text Summarizer Agent Demo")
    summarizer = TextSummarizerAgent()
    
    sample_text = """
    Artificial intelligence is transforming the way we work and live. It has applications in healthcare, 
    finance, transportation, and many other industries. Machine learning algorithms can process vast 
    amounts of data to identify patterns and make predictions. However, AI also raises important 
    ethical questions about privacy, bias, and the future of human employment. As AI continues to 
    evolve, society must carefully consider how to harness its benefits while mitigating potential risks.
    """
    
    print(f"\nAnalyzing text: {sample_text[:100]}...")
    result = await summarizer.execute(sample_text.strip())
    print(f"Summary: {result.data['summary']}")
    print(f"Statistics: {result.data['statistics']}")
    print(f"Readability: {result.data['readability']}")
    
    # Math Agent Demo
    print("\n\n3. Math Agent Demo")
    math_agent = MathAgent()
    
    math_operations = [
        {"operation": "add", "operands": [10, 20, 30]},
        {"operation": "multiply", "operands": [5, 4]},
        {"operation": "divide", "operands": [100, 4]},
        {"operation": "sqrt", "operands": [16]},
        "2 + 3 * 4"
    ]
    
    for i, operation in enumerate(math_operations, 1):
        print(f"\nMath Test {i}: {operation}")
        result = await math_agent.execute(operation)
        print(f"Result: {result.data}")
        print(f"Success: {result.success}")
    
    # Agent Status and Metrics
    print("\n\n4. Agent Status and Metrics")
    agents = [echo_agent, summarizer, math_agent]
    
    for agent in agents:
        print(f"\n{agent.metadata.name}:")
        print(f"Status: {agent.get_status()}")
        print(f"Metrics: {agent.get_metrics()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_agents())