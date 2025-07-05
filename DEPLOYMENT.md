# Deployment Guide - Vercel

This guide will help you deploy your sentiment analysis API to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm i -g vercel`
3. **GitHub Repository**: Your project should be pushed to GitHub

## Files Created for Deployment

- `app.py` - Flask API server
- `vercel.json` - Vercel configuration
- `static/index.html` - Web interface
- `requirements.txt` - Updated with Flask dependency

## Deployment Steps

### 1. Install Vercel CLI
```bash
npm install -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy from your project directory
```bash
vercel
```

### 4. Follow the prompts:
- Set up and deploy? → **Y**
- Which scope? → Select your account
- Link to existing project? → **N**
- Project name? → `sentiment-analysis-api` (or your preferred name)
- Directory? → **./** (current directory)
- Override settings? → **N**

### 5. Deploy to production
```bash
vercel --prod
```

## API Endpoints

Once deployed, your API will have these endpoints:

- **`/`** - Web interface for testing
- **`/api`** - API information
- **`/health`** - Health check
- **`/predict`** - Single text sentiment analysis
- **`/predict_batch`** - Batch sentiment analysis

## Testing the API

### Web Interface
Visit your deployed URL to use the web interface.

### API Testing
```bash
# Single prediction
curl -X POST https://your-app.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This feature is amazing! It works perfectly."}'

# Health check
curl https://your-app.vercel.app/health
```

## Environment Variables

If you need to add environment variables:
1. Go to your Vercel dashboard
2. Select your project
3. Go to Settings → Environment Variables
4. Add any required variables

## Troubleshooting

### Model Loading Issues
- Ensure `models/svm.pkl` is included in your repository
- Check that the model file is not too large (Vercel has size limits)

### Build Errors
- Check that all dependencies are in `requirements.txt`
- Ensure Python version compatibility

### API Errors
- Check the `/health` endpoint to verify model loading
- Review Vercel function logs in the dashboard

## Custom Domain (Optional)

1. Go to your Vercel dashboard
2. Select your project
3. Go to Settings → Domains
4. Add your custom domain

## Monitoring

- **Vercel Dashboard**: Monitor deployments and function logs
- **Analytics**: View usage statistics in the dashboard
- **Logs**: Check function execution logs for debugging

## Cost Considerations

- Vercel has a generous free tier
- Serverless functions have execution time limits
- Monitor usage to avoid unexpected charges

## Security Notes

- The API is public by default
- Consider adding authentication if needed
- Rate limiting is handled by Vercel
- Input validation is implemented in the API 