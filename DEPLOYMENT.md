# Deploying to Vercel

This guide will help you deploy your Flask sentiment analysis application to Vercel.

## Prerequisites

1. **Vercel CLI**: Install the Vercel CLI globally
   ```bash
   npm install -g vercel
   ```

2. **Vercel Account**: Sign up for a free account at [vercel.com](https://vercel.com)

## Deployment Steps

### 1. Install Vercel CLI (if not already installed)
```bash
npm install -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy the Application
From your project root directory, run:
```bash
vercel
```

### 4. Follow the Prompts
- **Set up and deploy**: Choose `Y`
- **Which scope**: Select your account
- **Link to existing project**: Choose `N` (for first deployment)
- **Project name**: Enter a name for your project (e.g., `sentiment-analysis-api`)
- **In which directory is your code located**: Press Enter (current directory)
- **Want to override the settings**: Choose `N`

### 5. Wait for Deployment
Vercel will:
- Upload your files
- Install dependencies
- Build the application
- Deploy to a production URL

### 6. Access Your Application
Once deployment is complete, you'll get:
- **Production URL**: `https://your-project-name.vercel.app`
- **Preview URL**: For testing changes

## Project Structure for Vercel

The deployment uses this structure:
```
├── api/
│   ├── index.py          # Main Flask application
│   └── requirements.txt  # Python dependencies
├── models/
│   └── svm.pkl          # Trained model
├── static/
│   └── index.html       # Frontend interface
├── vercel.json          # Vercel configuration
└── .vercelignore        # Files to exclude
```

## API Endpoints

After deployment, your API will be available at:
- **Homepage**: `https://your-project-name.vercel.app/`
- **API Info**: `https://your-project-name.vercel.app/api`
- **Health Check**: `https://your-project-name.vercel.app/health`
- **Predict**: `https://your-project-name.vercel.app/predict` (POST)
- **Batch Predict**: `https://your-project-name.vercel.app/predict_batch` (POST)

## Testing the Deployment

1. **Visit the homepage**: Open your production URL in a browser
2. **Test the API**: Use the web interface to analyze sentiment
3. **API testing**: Use tools like Postman or curl to test endpoints

### Example API Call
```bash
curl -X POST https://your-project-name.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This feature is amazing!"}'
```

## Environment Variables

If you need to add environment variables:
1. Go to your Vercel dashboard
2. Select your project
3. Go to Settings → Environment Variables
4. Add any required variables

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure `models/svm.pkl` is in the correct location
2. **Dependencies**: Check that all packages in `api/requirements.txt` are compatible
3. **Function Timeout**: The function timeout is set to 30 seconds in `vercel.json`

### Debugging

1. **Check Vercel logs**: Use `vercel logs` to view deployment logs
2. **Local testing**: Test locally with `vercel dev` before deploying
3. **Function logs**: Check function logs in the Vercel dashboard

## Updating the Deployment

To update your deployment:
```bash
vercel --prod
```

## Cost Considerations

- **Free Tier**: 100GB-hours per month
- **Serverless Functions**: Pay per execution
- **Bandwidth**: 100GB per month included

## Support

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/runtimes#official-runtimes/python)
- [Flask on Vercel](https://vercel.com/guides/flask)

## Monitoring

- **Vercel Dashboard**: Monitor deployments and function logs
- **Analytics**: View usage statistics in the dashboard
- **Logs**: Check function execution logs for debugging

## Security Notes

- The API is public by default
- Consider adding authentication if needed
- Rate limiting is handled by Vercel
- Input validation is implemented in the API 