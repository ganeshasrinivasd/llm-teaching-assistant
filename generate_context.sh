#!/bin/bash

# =============================================================================
# Context Generator for Claude
# Generates a summary of the current codebase for AI to understand
# =============================================================================

OUTPUT_FILE="claude_context.md"

echo "# Project Context for Claude" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Generated: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# -----------------------------------------------------------------------------
# 1. Project Structure
# -----------------------------------------------------------------------------
echo "## 1. Project Structure" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
find . -type f \( -name "*.py" -o -name "*.tsx" -o -name "*.ts" -o -name "*.json" \) \
  ! -path "./node_modules/*" \
  ! -path "./.git/*" \
  ! -path "./venv/*" \
  ! -path "./__pycache__/*" \
  ! -path "./dist/*" \
  ! -path "./build/*" \
  | head -100 >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# -----------------------------------------------------------------------------
# 2. Backend Files Content
# -----------------------------------------------------------------------------
echo "## 2. Backend Code" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Config
if [ -f "backend/core/config.py" ]; then
  echo "### backend/core/config.py" >> $OUTPUT_FILE
  echo '```python' >> $OUTPUT_FILE
  cat backend/core/config.py >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# Services
for file in backend/services/*.py; do
  if [ -f "$file" ]; then
    echo "### $file" >> $OUTPUT_FILE
    echo '```python' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# Routes
for file in backend/api/routes/*.py; do
  if [ -f "$file" ]; then
    echo "### $file" >> $OUTPUT_FILE
    echo '```python' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# Models
for file in backend/models/*.py; do
  if [ -f "$file" ]; then
    echo "### $file" >> $OUTPUT_FILE
    echo '```python' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# Main
if [ -f "backend/api/main.py" ]; then
  echo "### backend/api/main.py" >> $OUTPUT_FILE
  echo '```python' >> $OUTPUT_FILE
  cat backend/api/main.py >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 3. Frontend Key Files
# -----------------------------------------------------------------------------
echo "## 3. Frontend Code" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# App.tsx
if [ -f "frontend/src/App.tsx" ]; then
  echo "### frontend/src/App.tsx" >> $OUTPUT_FILE
  echo '```tsx' >> $OUTPUT_FILE
  cat frontend/src/App.tsx >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# API client
if [ -f "frontend/src/lib/api.ts" ]; then
  echo "### frontend/src/lib/api.ts" >> $OUTPUT_FILE
  echo '```typescript' >> $OUTPUT_FILE
  cat frontend/src/lib/api.ts >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# Components
for file in frontend/src/components/*.tsx; do
  if [ -f "$file" ]; then
    echo "### $file" >> $OUTPUT_FILE
    echo '```tsx' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# -----------------------------------------------------------------------------
# 4. Requirements & Dependencies
# -----------------------------------------------------------------------------
echo "## 4. Dependencies" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "backend/requirements.txt" ]; then
  echo "### backend/requirements.txt" >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  cat backend/requirements.txt >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

if [ -f "frontend/package.json" ]; then
  echo "### frontend/package.json" >> $OUTPUT_FILE
  echo '```json' >> $OUTPUT_FILE
  cat frontend/package.json >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 5. Environment Example
# -----------------------------------------------------------------------------
echo "## 5. Environment Variables" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "backend/.env.example" ]; then
  echo "### backend/.env.example" >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  cat backend/.env.example >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
elif [ -f "backend/.env" ]; then
  echo "### backend/.env (sanitized)" >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  cat backend/.env | sed 's/=.*/=<REDACTED>/' >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo "" >> $OUTPUT_FILE
echo "---" >> $OUTPUT_FILE
echo "## What I Need Claude To Do" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Implement v2 improvements:" >> $OUTPUT_FILE
echo "1. Add relevance threshold system (0.50/0.35 cutoffs)" >> $OUTPUT_FILE
echo "2. Add Semantic Scholar integration for dynamic paper fetching" >> $OUTPUT_FILE
echo "3. Add Pinecone service (optional, for production)" >> $OUTPUT_FILE
echo "4. Add Query Enhancement service (intent detection)" >> $OUTPUT_FILE
echo "5. Remove LeetCode feature" >> $OUTPUT_FILE
echo "6. Update teaching_service.py with new orchestration logic" >> $OUTPUT_FILE

echo ""
echo "✅ Context generated: $OUTPUT_FILE"
echo "📊 File size: $(wc -l < $OUTPUT_FILE) lines"
echo ""
echo "Upload this file to Claude!"