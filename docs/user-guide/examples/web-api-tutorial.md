# 🌐 Building a Web API with A3X

This tutorial shows how to build a complete REST API from scratch using A3X's autonomous capabilities.

## What We'll Build

We'll create a simple **Task Management API** with the following features:

- ✅ Create, read, update, and delete tasks
- ✅ Data validation and error handling
- ✅ JSON API responses
- ✅ Basic authentication
- ✅ Comprehensive tests
- ✅ API documentation

## Prerequisites

- ✅ A3X installed and configured
- ✅ API key set up
- ✅ Basic understanding of REST APIs

## Step 1: Project Initialization

Let's start by creating a proper Python project structure:

```bash
a3x run --goal "Create a Python FastAPI project with proper structure for a task management API" --config configs/sample.yaml
```

**What A3X will create:**
- Project directory structure
- `requirements.txt` with dependencies
- Basic FastAPI application setup
- Project configuration files

**Expected result:**
```
task-api/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── tests/              # Test directory
```

## Step 2: Task Data Model

Now let's create the task data model:

```bash
a3x run --goal "Create a Task model with id, title, description, status, created_at, and updated_at fields" --config configs/sample.yaml
```

**What A3X will create:**
- Pydantic models for Task
- Data validation
- Type hints and documentation

**Expected additions:**
```python
# In main.py or models.py
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.pending

class TaskCreate(TaskBase):
    pass

class Task(TaskBase):
    id: int
    created_at: datetime
    updated_at: datetime
```

## Step 3: API Endpoints

Let's implement the core CRUD operations:

```bash
a3x run --goal "Implement CRUD endpoints for tasks: GET /tasks, GET /tasks/{id}, POST /tasks, PUT /tasks/{id}, DELETE /tasks/{id}" --config configs/sample.yaml
```

**What A3X will create:**
- REST API endpoints
- Error handling
- Response models
- Status codes

**Expected endpoints:**
- `GET /tasks` - List all tasks
- `GET /tasks/{id}` - Get specific task
- `POST /tasks` - Create new task
- `PUT /tasks/{id}` - Update task
- `DELETE /tasks/{id}` - Delete task

## Step 4: Data Storage

Let's add a simple in-memory storage (for development):

```bash
a3x run --goal "Add in-memory storage for tasks with proper ID generation and data persistence during runtime" --config configs/sample.yaml
```

**What A3X will create:**
- Task storage service
- ID generation logic
- Data access layer

## Step 5: Authentication

Let's add basic authentication:

```bash
a3x run --goal "Add JWT token-based authentication to protect the API endpoints" --config configs/sample.yaml
```

**What A3X will create:**
- Authentication middleware
- JWT token generation and validation
- Login endpoint
- Protected route decorators

## Step 6: Testing

Let's create comprehensive tests:

```bash
a3x run --goal "Create comprehensive test suite covering all endpoints, authentication, and error cases" --config configs/sample.yaml
```

**What A3X will create:**
- Unit tests for all endpoints
- Authentication tests
- Error handling tests
- Test utilities and fixtures

## Step 7: API Documentation

Let's generate API documentation:

```bash
a3x run --goal "Generate OpenAPI/Swagger documentation for the task management API" --config configs/sample.yaml
```

**What A3X will create:**
- Automatic API documentation
- Interactive Swagger UI
- Request/response examples

## Running the Application

Now let's test our complete API:

```bash
# Run the application
uvicorn main:app --reload

# In another terminal, test the API
curl http://localhost:8000/tasks/
curl -X POST "http://localhost:8000/tasks/" \
     -H "Content-Type: application/json" \
     -d '{"title": "Test Task", "description": "This is a test"}'

# Run tests
pytest tests/
```

## Expected Final Structure

```
task-api/
├── main.py              # Main FastAPI application
├── models.py           # Pydantic models
├── services/           # Business logic
│   └── task_service.py
├── auth/              # Authentication
│   └── jwt_handler.py
├── tests/             # Test suite
│   ├── test_tasks.py
│   └── test_auth.py
├── requirements.txt   # Dependencies
├── README.md         # Documentation
└── .gitignore        # Git ignore file
```

## What You've Learned

In this tutorial, you saw A3X:

1. **Initialize a complete project** structure
2. **Design and implement** data models
3. **Build REST API** endpoints
4. **Add authentication** and security
5. **Create comprehensive tests**
6. **Generate documentation** automatically

## Next Steps

Try these variations:

**Add database persistence:**
```bash
a3x run --goal "Replace in-memory storage with SQLite database" --config configs/sample.yaml
```

**Add advanced features:**
```bash
a3x run --goal "Add task categories, priorities, and due dates" --config configs/sample.yaml
```

**Deploy the API:**
```bash
a3x run --goal "Create Docker configuration for API deployment" --config configs/sample.yaml
```

## Troubleshooting

**Common issues:**

1. **Import errors**: Make sure all dependencies are installed
2. **Authentication failures**: Check JWT configuration
3. **CORS issues**: Configure CORS for frontend access

**Getting help:**
- Check the generated code for obvious issues
- Review error messages carefully
- Ask A3X to fix specific problems

---

**🎉 Congratulations!** You've built a complete REST API using autonomous coding. This demonstrates A3X's ability to handle complex, multi-step projects with proper architecture and testing.